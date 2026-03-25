import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
import wandb

class CustomSmoothL1Loss(nn.Module):
    def __init__(self):
        super(CustomSmoothL1Loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, input, target, mask):
        # Smooth L1 Loss 계산
        loss = self.smooth_l1_loss(input, target)
        
        # 마스크 적용
        masked_loss = loss * mask
        loss_sum = masked_loss.sum()
        num_valid = mask.sum()
        if num_valid == 0:
            return loss_sum * 0.0
        return loss_sum / num_valid
    
class CustomConsistencyLoss(nn.Module):
    def __init__(self):
        super(CustomConsistencyLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, curr_heightmap, new_roi, mask):
        """
        curr_heightmap: (B, 1, 200, 48)  # GT heightmap
        new_roi: (B, 3, 200, 48)         # [y_pos, x_pos, value]
          - new_roi[:, 0, :, :]: longitudinal
          - new_roi[:, 1, :, :]: lateral
          - new_roi[:, 2, :, :]: 예측된 height 값
        mask: (B, 1, 200, 48)            # 유효 영역 마스크
        """
        B, _, H, W = curr_heightmap.shape

        # 좌표 변환
        y_indices = new_roi[:, 0, :, :].long()  # (B, 200, 48)
        x_indices = new_roi[:, 1, :, :].long()   # (B, 200, 48)

        # 유효 범위 체크 (0 ≤ y < 200, 0 ≤ x < 48)
        valid_mask = (y_indices >= 0) & (y_indices < H) & (x_indices >= 0) & (x_indices < W)

        # 유효한 좌표만 필터링
        batch_indices = torch.arange(B, device=curr_heightmap.device).view(B, 1, 1).expand_as(valid_mask)  # (B, H, W)
        valid_y = y_indices[valid_mask]  # (N,) #종방향 index 
        valid_x = x_indices[valid_mask]  # (N,)
        valid_new_roi_2 = new_roi[:, 2, :, :][valid_mask]  # (N,)
        valid_batch_indices = batch_indices[valid_mask]

        # GT heightmap에서 해당 위치의 값 추출
        sampled_curr_heightmap = curr_heightmap[valid_batch_indices, 0, valid_y, valid_x]  # (N,)
        # Smooth L1 Loss 계산
        loss = self.l1_loss(sampled_curr_heightmap, valid_new_roi_2)  # (N,)

        # 마스크 적용
        sampled_mask = mask[valid_batch_indices, 0, valid_y, valid_x]  # (N,)
        masked_loss = loss * sampled_mask
        
        loss_sum = masked_loss.sum()
        num_valid = sampled_mask.sum()
        epsilon = 1e-6
        return loss_sum / (num_valid + epsilon)


class reproject_prev_heightmap(nn.Module):
    def __init__(self):
        super(reproject_prev_heightmap, self).__init__()   
        self.maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
                                        [-1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 0, 1]], dtype=float)
        self.meter_per_pixel = 0.5
        self.R_vg = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]], dtype=float)
        self.R_gc = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]], dtype=float)
        self.grid_y, self.grid_x = torch.meshgrid(torch.linspace(0, 199, 200),
                                        torch.linspace(0, 47,48), indexing='ij')
        self.grid_y = self.grid_y / 199 * 2 - 1  # (H, W)
        self.grid_x = self.grid_x / 47 * 2 - 1  # (H, W)
        
        self.grid = torch.stack((self.grid_x, self.grid_y), dim=-1).unsqueeze(0)  # shape: (1, H, W, 2)
    
    def forward(self, prev_heightmap, vehicle_pose_prev, vehicle_pose_curr, extrinsic_prev, extrinsic_curr, prev_road2cam, curr_road2cam):
        B, _, bevH, bevW = prev_heightmap.shape  # B x 1 x H x W

        # 기존 xv_real, yv_real을 torch.Tensor로 변환 (broadcasting을 위해 view 사용)
        device = prev_heightmap.device
        xv_real = torch.arange(bevH, device=device).view(1, bevH, 1).expand(B, bevH, bevW)
        yv_real = torch.arange(bevW, device=device).view(1, 1, bevW).expand(B, bevH, bevW)
        
        xv_real = 103 - ((xv_real + 0.5) / 2)
        yv_real = (yv_real / 2) - 12
        zv_real = prev_heightmap.squeeze(1)  # (B, H, W)
            
        ### Prev BEV coord making ###
        prev_bev_coords = torch.stack([yv_real, xv_real, zv_real], dim=1)  # (B, 3, H, W)
        prev_bev_coords = prev_bev_coords.view(B, 3, -1)  # (B, 3, HW)
        ones = torch.ones(B, 1, prev_bev_coords.shape[2], device=device)
        prev_bev_coords = torch.cat([prev_bev_coords, ones], dim=1)  # (B, 4, HW)

        ### Prev BEV coord to Prev waymo camera coord ###
        prev_cam_coords = torch.bmm(prev_road2cam, prev_bev_coords)  # (B, 4, HW)  #횡 종 z : z.min -2.3732 z.max -0.55
        
        ### prev camera coord to global coord ###
        prev_vehicle_coords = torch.bmm(extrinsic_prev, prev_cam_coords)  # (B, 4, HW)
        global_coords = torch.bmm(vehicle_pose_prev, prev_vehicle_coords)  # (B, 4, HW)

        ### Global coord to curr vehicle coord ###
        curr_R = vehicle_pose_curr[:, :3, :3]  # (B, 3, 3)
        curr_T = vehicle_pose_curr[:, :3, 3].unsqueeze(-1)  # (B, 3, 1)
        curr_vehicle_coords = torch.bmm(curr_R.transpose(1, 2), global_coords[:, :3, :]) - torch.bmm(curr_R.transpose(1, 2), curr_T)  # (B, 3, HW)

        ### Global coord to curr camera coord ###
        R_ex = extrinsic_curr[:, :3, :3]  # (B, 3, 3)
        T_ex = extrinsic_curr[:, :3, 3].unsqueeze(-1)  # (B, 3, 1)

        # Translate & rotate point cloud to waymo cam position
        prev_to_curr_points_cam = torch.bmm(R_ex.transpose(1, 2), curr_vehicle_coords) - torch.bmm(R_ex.transpose(1, 2), T_ex)  # (B, 3, HW)
        filtered_prev_to_curr_points_cam = torch.cat([prev_to_curr_points_cam, torch.ones(B, 1, prev_to_curr_points_cam.shape[2], device=device)], dim=1)  # (B, 4, HW)

        # waymo camera coord to current road coord
        curr_lane2persformer = torch.inverse(curr_road2cam.cpu()).to(device)  # (B, 4, 4)
        new_roi = torch.bmm(curr_lane2persformer, filtered_prev_to_curr_points_cam) # (B, 4, HW)
        ipm_roi = torch.stack([206 - 2 * new_roi[:, 1, :], 24 + 2 * new_roi[:, 0, :], new_roi[:, 2, :]], dim=1)  # (B, 3, HW)

        ipm_roi = ipm_roi.view(B, 3, bevH, bevW)  # (B, 3, H, W)
        return ipm_roi
    
    
class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.emb_loss = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.L1Loss(reduction="sum")
        self.bce_loss = nn.BCELoss()
        self.l1_loss = CustomSmoothL1Loss()
        self.reprojector = reproject_prev_heightmap()
        self.consis_loss = CustomConsistencyLoss()

        self.max_x = 103
        self.meter_per_pixel = 0.5


    def forward(self, inputs, prev_input, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, gt_height = None, intrinsic= None, prev_intrinsic =None, road2cam = None, extrinsic= None,
                cam_w_ext=None, prev_cam_w_ext=None, vehicle_pose=None, prev_vehicle_pose=None, prev_road2cam=None, train=True):
        
        (pred, emb, offset_y), heightmap, (pred_2d, emb_2d), anchor_weighted_height = self.model(inputs, intrinsic, road2cam, prev=False)
        prev_heightmap = self.model(prev_input, prev_intrinsic, prev_road2cam, prev=True)
        reprojected_prev_ipm = self.reprojector(prev_heightmap, prev_vehicle_pose, vehicle_pose, prev_cam_w_ext, cam_w_ext, prev_road2cam, road2cam)

        ## 3d
        gt_heightmap = gt_height[:,0,:,:].unsqueeze(1)
        gt_heightmask = gt_height[:,1,:,:].unsqueeze(1)
        loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)
        loss_emb = self.emb_loss(emb, gt_instance)
        loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)
        loss_height = self.l1_loss(heightmap, gt_heightmap, gt_heightmask)
        loss_consistency = self.consis_loss(gt_heightmap, reprojected_prev_ipm, gt_heightmask)
        
        loss_total = 5 * loss_seg + loss_emb
        loss_total = loss_total.unsqueeze(0)
        loss_offset = 60 * loss_offset.unsqueeze(0)
        loss_height = 10 * loss_height.unsqueeze(0)
        loss_consistency = 3 * loss_consistency.unsqueeze(0)
        

        ## auxiliary: anchor weighted sum supervised by gt heightmap (only when use_img=True)
        if anchor_weighted_height is not None:
            loss_aux = 5 * self.l1_loss(anchor_weighted_height, gt_heightmap, gt_heightmask).unsqueeze(0)
        else:
            loss_aux = torch.zeros(1, device=gt_seg.device)

        loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
        loss_emb_2d = self.emb_loss(emb_2d, image_gt_instance)
        loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
        loss_total_2d = loss_total_2d.unsqueeze(0)

        return pred, loss_total, loss_offset, loss_total_2d, loss_height, loss_consistency, loss_aux


def train_epoch(model, dataset, optimizer, configs, epoch):
    # Last iter as mean loss of whole epoch
    model.train()
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    input_data, prev_input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance, intrinsic, prev_intrinsic, extrinsic, road2cam, heightmap \
        , cam_w_extrinsics, prev_cam_w_extrinsics, vehicle_pose, prev_vehicle_pose, prev_extrinsic_road2cam) in enumerate(dataset):
        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)]
        input_data = input_data.to(rank, non_blocking=True)
        prev_input_data = prev_input_data.to(rank, non_blocking=True)
        
        image_intrinsic = intrinsic.to(rank, non_blocking=True)
        prev_image_intrinsic = prev_intrinsic.to(rank, non_blocking=True)
        
        image_extrinsic = extrinsic.to(rank, non_blocking=True)
        image_road2cam = road2cam.to(rank, non_blocking=True)
        
        gt_heightmap = heightmap.to(rank, non_blocking=True)
        gt_seg_data = gt_seg_data.to(rank, non_blocking=True)
        gt_emb_data = gt_emb_data.to(rank, non_blocking=True)
        offset_y_data = offset_y_data.to(rank, non_blocking=True)
        z_data = z_data.to(rank, non_blocking=True)
        image_gt_segment = image_gt_segment.to(rank, non_blocking=True)
        image_gt_instance = image_gt_instance.to(rank, non_blocking=True)
        
        ''' for consistency loss'''
        
        cam_w_extrinsics = cam_w_extrinsics.to(rank, non_blocking=True)
        prev_cam_w_extrinsics = prev_cam_w_extrinsics.to(rank, non_blocking=True)
        vehicle_pose = vehicle_pose.to(rank, non_blocking=True)
        prev_vehicle_pose = prev_vehicle_pose.to(rank, non_blocking=True)
        prev_extrinsic_road2cam = prev_extrinsic_road2cam.to(rank, non_blocking=True)
        
        prediction, loss_total_bev, loss_offset, loss_total_2d, loss_height, loss_consistency, loss_aux= model(input_data, prev_input_data, gt_seg_data,
                                                                                gt_emb_data,
                                                                                offset_y_data, z_data,
                                                                                image_gt_segment,
                                                                                image_gt_instance,
                                                                                gt_heightmap,
                                                                                intrinsic=image_intrinsic,
                                                                                prev_intrinsic=prev_image_intrinsic,
                                                                                extrinsic = image_extrinsic,
                                                                                road2cam = image_road2cam,
                                                                                cam_w_ext=cam_w_extrinsics,
                                                                                prev_cam_w_ext=prev_cam_w_extrinsics,
                                                                                vehicle_pose=vehicle_pose,
                                                                                prev_vehicle_pose=prev_vehicle_pose,
                                                                                prev_road2cam=prev_extrinsic_road2cam,
                                                                                train=True)
        loss_back_bev = loss_total_bev.mean()
        loss_back_2d = loss_total_2d.mean()
        loss_offset = loss_offset.mean()
        loss_height = loss_height.mean()
        loss_aux = loss_aux.mean()
        loss_consistency = loss_consistency.mean()

        loss_back_total = loss_back_bev + loss_offset + 0.5 * loss_back_2d + loss_height + loss_consistency# + loss_aux
        ''' caclute loss '''

        optimizer.zero_grad()
        loss_back_total.backward()
        optimizer.step()
        wandb.log({"loss_back_bev": loss_back_bev.item(), "loss_offset": loss_offset.item(),
                   "loss_height": loss_height.item(), "loss_aux": loss_aux.item(),
                   "loss_back_total": loss_back_total.item(), "loss_back_2d": loss_back_2d.item()
         })
        if idx % 50 == 0:
            print(idx, loss_back_bev.item(), '*' * 10)
        if idx % 300 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            loss_iter = {"BEV Loss": loss_back_bev.item(), 'offset loss': loss_offset.item(),'height loss': loss_height.item()
                            ,"F1_BEV_seg": f1_bev_seg}
            wandb.log({"F1-bev-seg": f1_bev_seg})


def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is'+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)
    wandb.init(project="sc-lane")
    ''' models and optimizer '''
    model = configs.model()
    model = Combine_Model_and_Loss(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler)
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    ''' dataset '''
    Dataset = getattr(configs, "train_dataset", None)
    if Dataset is None:
        Dataset = configs.training_dataset
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True)

    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean)
    #         return
    torch.cuda.empty_cache()
    for epoch in range(configs.epochs):
        print_epoch = epoch
        # print('*' * 100, print_epoch)
        
        train_epoch(model, train_loader, optimizer, configs, print_epoch)
        scheduler.step()
        save_model_dp(model, optimizer, configs.model_save_path, 'ep%03d.pth' % print_epoch)
        save_model_dp(model, None, configs.model_save_path, 'latest.pth')


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('tools/sc_lane_config.py', gpu_id=[0])
