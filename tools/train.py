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
        return loss_sum / num_valid

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

        self.max_x = 103
        self.meter_per_pixel = 0.5


    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, gt_height = None, intrinsic= None, extrinsic= None, road2cam = None, train=True):
        res = self.model(inputs, intrinsic, road2cam)
        pred, emb, offset_y = res[0]
        heightmap = res[1]
        pred_2d, emb_2d = res[2]
        anchor_weighted_height = res[3]

        gt_heightmap = gt_height[:,0,:,:].unsqueeze(1)
        gt_heightmask = gt_height[:,1,:,:].unsqueeze(1)

        if train:
            ## 3d
            loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)
            loss_emb = self.emb_loss(emb, gt_instance)
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)
            loss_height = self.l1_loss(heightmap, gt_heightmap, gt_heightmask)
            loss_total = 5 * loss_seg + loss_emb
            loss_total = loss_total.unsqueeze(0)
            loss_offset = 60 * loss_offset.unsqueeze(0)
            loss_height = 10 * loss_height.unsqueeze(0)

            ## auxiliary: anchor weighted sum supervised by gt heightmap (only when use_img=True)
            if anchor_weighted_height is not None:
                loss_aux = 5 * self.l1_loss(anchor_weighted_height, gt_heightmap, gt_heightmask).unsqueeze(0)
            else:
                loss_aux = torch.zeros(1, device=gt_seg.device)

            loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
            loss_emb_2d = self.emb_loss(emb_2d, image_gt_instance)
            loss_total_2d = 5 * loss_seg_2d + 0.5 * loss_emb_2d
            loss_total_2d = loss_total_2d.unsqueeze(0)

            return pred, loss_total, loss_offset, loss_total_2d, loss_height, loss_aux
        else:
            return pred


def train_epoch(model, dataset, optimizer, configs, epoch):
    # Last iter as mean loss of whole epoch
    model.train()
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance, intrinsic, extrinsic, road2cam, heightmap) in enumerate(
            dataset):
        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        # pdb.set_trace()
        input_data = input_data.cuda()
        image_intrinsic = intrinsic.cuda()
        image_extrinsic = extrinsic.cuda()
        image_road2cam = road2cam.cuda()
        gt_heightmap = heightmap.cuda()
        gt_seg_data = gt_seg_data.cuda()
        gt_emb_data = gt_emb_data.cuda()
        offset_y_data = offset_y_data.cuda()
        z_data = z_data.cuda()
        image_gt_segment = image_gt_segment.cuda()
        image_gt_instance = image_gt_instance.cuda()
        #prediction, loss_total_bev, loss_offset, loss_z, loss_total_2d , loss_height, loss_backproj= model(input_data, gt_seg_data,
        prediction, loss_total_bev, loss_offset, loss_total_2d, loss_height, loss_aux = model(
                                                                                input_data, gt_seg_data,
                                                                                gt_emb_data,
                                                                                offset_y_data, z_data,
                                                                                image_gt_segment,
                                                                                image_gt_instance,
                                                                                gt_heightmap,
                                                                                intrinsic=image_intrinsic,
                                                                                extrinsic=image_extrinsic,
                                                                                road2cam=image_road2cam)
        loss_back_bev = loss_total_bev.mean()
        loss_back_2d = loss_total_2d.mean()
        loss_offset = loss_offset.mean()
        loss_height = loss_height.mean()
        loss_aux = loss_aux.mean()

        loss_back_total = loss_back_bev + loss_offset + 0.5 * loss_back_2d + loss_height + loss_aux
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
