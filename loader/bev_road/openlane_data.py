import copy
import json
import os

import cv2
import numpy as np
import torch
import pickle
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from utils.coord_util import ego2image,IPM2ego_matrix, ego2image_filtered
from functools import cmp_to_key
from scipy.interpolate import griddata

class OpenLane_dataset_with_offset(Dataset):
    def __init__(self, image_paths,
                   gt_paths,
                   map_paths,
                   x_range,
                   y_range,
                   meter_per_pixel,
                   data_trans,
                   output_2d_shape,
                   input_shape=(600, 800)):
        self.heatmap_h = 18
        self.heatmap_w = 32
        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.map_paths = map_paths
        self.cnt_list = []
        self.lane3d_thick = 1
        self.lane2d_thick = 3
        self.lane_length_threshold = 3  #
        card_list = os.listdir(self.gt_paths)
        for card in card_list:
            gt_paths = os.path.join(self.gt_paths, card)
            gt_list = sorted(os.listdir(gt_paths))  # 시간순 정렬
            for cnt in gt_list:
                self.cnt_list.append([card, cnt])

        # prev frame이 같은 scene에 존재하는 index만 유효
        self.valid_indices = []
        for i in range(1, len(self.cnt_list)):
            if self.cnt_list[i][0] == self.cnt_list[i-1][0]:
                self.valid_indices.append(i)

        self.sep_map_paths = "/media/vdcl/T9/Waymo/map_data_training_seperate"
        ''' transform loader '''
        self.output2d_size = output_2d_shape
        self.trans_image = data_trans
        self.input_h, self.input_w = input_shape
        self.ipm_h, self.ipm_w = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int(
            (self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)
        self.matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel), int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)
        
    def project_bev_height_map_to_image_plane(self, bev_height_map, extrinsic_matrix, intrinsic_matrix):
        # Create homogeneous coordinates for height map points
        points_3d = np.ones((self.ipm_h *  self.ipm_w, 4))
        points_3d[:, :2] = np.mgrid[self.x_range[0]:self.x_range[0] + self.ipm_h, self.y_range[0]*2:self.y_range[1]*2].T.reshape(-1, 2)*self.meter_per_pixel
        # Reshape height map to 1D array
        points_3d[:,2] = bev_height_map.flatten()

        # Project 3D points to image plane
        image_points = intrinsic_matrix @ extrinsic_matrix[:3,:] @ points_3d.T

        # Normalize image points
        image_points_normalized = image_points[:2] / image_points[2]
        # print(image_points_normalized)
        return image_points_normalized

    
    def bev2ipm(self, bev, matrix_IPM2ego):
        ego_points = np.array([bev[0], bev[1]])
        ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (ego_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
        ipm_points_ = np.zeros_like(ipm_points)
        ipm_points_[0] = ipm_points[1]
        ipm_points_[1] = ipm_points[0]
        res_points = np.concatenate([ipm_points, np.array([bev[2]])], axis=0)
        return res_points    
    
    def get_y_offset_and_z(self, res_d):
        '''
        :param res_d: res_d
        :param instance_seg:
        :return:
        '''

        def caculate_distance(base_points, lane_points, lane_z, lane_points_set):
            '''
            :param base_points: base_points n * 2
            :param lane_points: # res_lane_points[idx] = np.array([base_points, y_points])  
            :return:
            '''
            condition = np.where(
                (lane_points_set[0] == int(base_points[0])) & (lane_points_set[1] == int(base_points[1])))
            if len(condition[0]) == 0:
                return None, None
            lane_points_selected = lane_points.T[condition]  #
            lane_z_selected = lane_z.T[condition]
            offset_y = np.mean(lane_points_selected[:, 1]) - base_points[1]
            z = np.mean(lane_z_selected[:, 1])
            return offset_y, z

        # instance_seg = np.zeros((450, 120), dtype=np.uint8)
        res_lane_points = {}
        res_lane_points_z = {}
        res_lane_points_bin = {}
        res_lane_points_set = {}
        for idx in res_d:
            ipm_points_ = np.array(res_d[idx])
            ipm_points = ipm_points_.T[np.where((ipm_points_[1] >= 0) & (ipm_points_[1] < self.ipm_h))].T  #
            if len(ipm_points[0]) <= 1:
                continue
            x, y, z = ipm_points[1], ipm_points[0], ipm_points[2]
            base_points = np.linspace(x.min(), x.max(),
                                      int((x.max() - x.min()) // 0.05))
            base_points_bin = np.linspace(int(x.min()), int(x.max()),
                                          int(int(x.max()) - int(x.min())) + 1)  # .astype(np.int)
            if len(x) <= 1:
                continue
            elif len(x) <= 2:
                function1 = interp1d(x, y, kind='linear',
                                     fill_value="extrapolate")  #
                function2 = interp1d(x, z, kind='linear')
            elif len(x) <= 3:
                function1 = interp1d(x, y, fill_value="extrapolate")
                function2 = interp1d(x, z)
            else:
                function1 = interp1d(x, y, fill_value="extrapolate")
                function2 = interp1d(x, z)
            y_points = function1(base_points)
            y_points_bin = function1(base_points_bin)
            z_points = function2(base_points)
            res_lane_points[idx] = np.array([base_points, y_points])  #
            res_lane_points_z[idx] = np.array([base_points, z_points])
            res_lane_points_bin[idx] = np.array([base_points_bin, y_points_bin]).astype(int)
            res_lane_points_set[idx] = np.array([base_points, y_points]).astype(
                int)

        offset_map = np.zeros((self.ipm_h, self.ipm_w))
        z_map = np.zeros((self.ipm_h, self.ipm_w))
        ipm_image = np.zeros((self.ipm_h, self.ipm_w))
        for idx in res_lane_points_bin:
            lane_bin = res_lane_points_bin[idx].T
            for point in lane_bin:
                row, col = point[0], point[1]
                if not (0 < row < self.ipm_h and 0 < col < self.ipm_w):  #
                    continue
                ipm_image[row, col] = idx
                center = np.array([row, col])
                offset_y, z = caculate_distance(center, res_lane_points[idx], res_lane_points_z[idx],
                                                res_lane_points_set[idx])  #
                if offset_y is None:  #
                    ipm_image[row, col] = 0
                    continue
                if offset_y > 1:
                    offset_y = 1
                if offset_y < 0:
                    offset_y = 0
                offset_map[row][col] = np.clip(offset_y, 0, 1)
                z_map[row][col] = z

        return ipm_image, offset_map, z_map

    def get_seg_offset(self, idx, smooth=False):
        gt_path = os.path.join(self.gt_paths, self.cnt_list[idx][0], self.cnt_list[idx][1])
        prev_gt_path = os.path.join(self.gt_paths, self.cnt_list[idx-1][0], self.cnt_list[idx-1][1])
        
        heightmap_path = os.path.join(self.map_paths, self.cnt_list[idx][0], self.cnt_list[idx][1].replace('json', 'npy'))
        
        image_path = os.path.join(self.image_paths, self.cnt_list[idx][0], self.cnt_list[idx][1].replace('json', 'jpg'))
        prev_image_path = os.path.join(self.image_paths, self.cnt_list[idx-1][0], self.cnt_list[idx-1][1].replace('json', 'jpg'))
        
        image = cv2.imread(image_path)
        prev_image = cv2.imread(prev_image_path)
        image_h, image_w, _ = image.shape
        
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        with open(heightmap_path, 'rb') as map_f:
            bev_height_map_mask = np.load(map_f)
    
        with open(prev_gt_path, 'r') as f:
            prev_gt = json.load(f)
        
        vehicle_pose = np.array(gt['pose'])
        prev_vehicle_pose = np.array(prev_gt['pose'])
        cam_w_extrinsics = np.array(gt['extrinsic'])
        prev_cam_w_extrinsics = np.array(prev_gt['extrinsic'])

        maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
                                            [-1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 0, 1]], dtype=float)
        
        cam_extrinsics = cam_w_extrinsics @ maxtrix_camera2camera_w  #
            
        R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        
        cam_extrinsics_persformer = copy.deepcopy(cam_w_extrinsics)
        cam_extrinsics_persformer[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), cam_extrinsics_persformer[:3, :3]),
            R_vg), R_gc)
        cam_extrinsics_persformer[0:2, 3] = 0.0
        matrix_lane2persformer = cam_extrinsics_persformer @ np.linalg.inv(maxtrix_camera2camera_w)

        prev_cam_extrinsics_persformer = copy.deepcopy(prev_cam_w_extrinsics)
        prev_cam_extrinsics_persformer[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), prev_cam_extrinsics_persformer[:3, :3]),
            R_vg), R_gc)
        prev_cam_extrinsics_persformer[0:2, 3] = 0.0
        prev_matrix_lane2persformer = prev_cam_extrinsics_persformer @ np.linalg.inv(maxtrix_camera2camera_w)

        cam_intrinsic = np.array(gt['intrinsic'])
        prev_cam_intrinsic = np.array(prev_gt['intrinsic'])
        
        lanes = gt['lane_lines']
        image_gt = np.zeros((image_h, image_w), dtype=np.uint8)
        res_points_d = {}
        for idx in range(len(lanes)):
            lane1 = lanes[idx]
            lane_camera_w = np.array(lane1['xyz']).T[np.array(lane1['visibility']) >= 1.0].T
            lane_camera_w = np.vstack((lane_camera_w, np.ones((1, lane_camera_w.shape[1]))))
            lane_ego_persformer = matrix_lane2persformer @ lane_camera_w  #
            lane_ego_persformer[0], lane_ego_persformer[1] = lane_ego_persformer[1], -1 * lane_ego_persformer[0]
            lane_ego = cam_w_extrinsics @ lane_camera_w  #
            ''' plot uv '''
            uv1 = ego2image(lane_ego[:3], cam_intrinsic, cam_extrinsics)
            
            cv2.polylines(image_gt, [uv1[0:2, :].T.astype(int)], False, idx + 1, self.lane2d_thick)

            distance = np.sqrt((lane_ego_persformer[1][0] - lane_ego_persformer[1][-1]) ** 2 + (
                    lane_ego_persformer[0][0] - lane_ego_persformer[0][-1]) ** 2)
            if distance < self.lane_length_threshold:
                continue
            
            y = lane_ego_persformer[1]
            x = lane_ego_persformer[0]
            z = lane_ego_persformer[2]

            if smooth:
                if len(x) < 2:
                    continue
                elif len(x) == 2:
                    curve = np.polyfit(x, y, 1)
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) == 3:
                    curve = np.polyfit(x, y, 2)
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    curve = np.polyfit(x, y, 3)
                    function2 = interp1d(x, z, kind='cubic')
                x_base = np.linspace(min(x), max(x), 20)
                y_pred = np.poly1d(curve)(x_base)
                ego_points = np.array([x_base, y_pred])
                z = function2(x_base)
            else:
                ego_points = np.array([x, y])

            ipm_points = np.linalg.inv(self.matrix_IPM2ego[:, :2]) @ (ego_points[:2] - self.matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)
            res_points_d[idx + 1] = res_points
        ipm_gt, offset_y_map, z_map = self.get_y_offset_and_z(res_points_d)

        return image, prev_image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, prev_cam_intrinsic, bev_height_map_mask, \
            matrix_lane2persformer, prev_matrix_lane2persformer, cam_w_extrinsics, prev_cam_w_extrinsics, vehicle_pose, prev_vehicle_pose
    
    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        real_idx = self.valid_indices[idx]
        image, prev_image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, prev_cam_intrinsic, bev_height_map_mask, \
            cam2road, prev_cam2road, cam_w_extrinsics, prev_cam_w_extrinsics, vehicle_pose, prev_vehicle_pose = self.get_seg_offset(real_idx)
        image_h, image_w, _ = image.shape
        
        cam_intrinsic[0] *= self.input_w / image_w
        cam_intrinsic[1] *= self.input_h/ image_h
        
        prev_cam_intrinsic[0] *= self.input_w / image_w
        prev_cam_intrinsic[1] *= self.input_h / image_h
        
        transformed = self.trans_image(image=image)
        prev_transformed = self.trans_image(image=prev_image)
        
        image = transformed["image"]
        prev_image = prev_transformed["image"]
        
        intrinsic = torch.tensor(cam_intrinsic)
        prev_intrinsic = torch.tensor(prev_cam_intrinsic)
        extrinsic = torch.tensor(cam_extrinsics)
        
        cam_w_extrinsics = torch.tensor(cam_w_extrinsics)
        prev_cam_w_extrinsics = torch.tensor(prev_cam_w_extrinsics)
        
        vehicle_pose = torch.tensor(vehicle_pose)   
        prev_vehicle_pose = torch.tensor(prev_vehicle_pose)
        
        road2cam = np.linalg.inv(cam2road)
        prev_road2cam = np.linalg.inv(prev_cam2road)
        
        extrinsic_road2cam = torch.tensor(road2cam)
        prev_extrinsic_road2cam = torch.tensor(prev_road2cam)
        ''' 2d gt'''
        image_gt = cv2.resize(image_gt, (self.output2d_size[1], self.output2d_size[0]), interpolation=cv2.INTER_NEAREST)
        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # h, w, c
        image_gt_segment = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1
        
        ''' 3d gt'''
        ipm_gt_instance = torch.tensor(ipm_gt).unsqueeze(0)  # h, w, c0
        ipm_gt_offset = torch.tensor(offset_y_map).unsqueeze(0)
        ipm_gt_z = torch.tensor(z_map).unsqueeze(0)
        ipm_gt_segment = torch.clone(ipm_gt_instance)
        ipm_gt_segment[ipm_gt_segment > 0] = 1
        image_gt_heightmap = torch.tensor(bev_height_map_mask)

        return image, prev_image, ipm_gt_segment.float(), ipm_gt_instance.float(), ipm_gt_offset.float(), ipm_gt_z.float(), image_gt_segment.float(), image_gt_instance.float(), intrinsic.float(), prev_intrinsic.float(), extrinsic.float(), extrinsic_road2cam.float(), image_gt_heightmap.float() \
            , cam_w_extrinsics.float(), prev_cam_w_extrinsics.float(), vehicle_pose.float(), prev_vehicle_pose.float(), prev_extrinsic_road2cam.float()

    def __len__(self):
        return len(self.valid_indices)


class OpenLane_dataset_with_offset_val(Dataset):
    def __init__(self, image_paths,
                 gt_paths,
                 map_paths,
                 data_trans,
                 x_range=(3, 103),
                 y_range=(-12, 12),
                 meter_per_pixel=0.5,
                 input_shape=(600, 800)):
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.map_paths = map_paths

        ''' get all list '''
        self.cnt_list = []
        card_list = os.listdir(self.gt_paths)
        for card in card_list:
            gt_paths = os.path.join(self.gt_paths, card)
            gt_list = os.listdir(gt_paths)
            for cnt in gt_list:
                self.cnt_list.append([card, cnt])

        self.sep_map_paths = "/media/vdcl/T9/Waymo/map_data_validation"
        ''' transform loader '''
        self.trans_image = data_trans
        self.x_range = list(x_range)
        self.y_range = list(y_range)
        self.meter_per_pixel = meter_per_pixel
        self.input_h, self.input_w = input_shape

    def bev2ipm(self, bev, matrix_IPM2ego):
        ego_points = np.array([bev[0], bev[1]])
        ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (ego_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
        ipm_points_ = np.zeros_like(ipm_points)
        ipm_points_[0] = ipm_points[1]
        ipm_points_[1] = ipm_points[0]
        res_points = np.concatenate([ipm_points, np.array([bev[2]])], axis=0)
        return res_points    

    def __getitem__(self, idx):
        '''get image '''
        gt_path = os.path.join(self.gt_paths, self.cnt_list[idx][0], self.cnt_list[idx][1])
        image_path = os.path.join(self.image_paths, self.cnt_list[idx][0], self.cnt_list[idx][1].replace('json', 'jpg'))
        heightmap_path = os.path.join(self.map_paths, self.cnt_list[idx][0], self.cnt_list[idx][1].replace('json', 'npy'))
        with open(heightmap_path, 'rb') as map_f:
            bev_height_map_mask = np.load(map_f)
        image = cv2.imread(image_path)
        image_original = image
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        
        cam_w_extrinsics = np.array(gt['extrinsic'])  # 4x4 shape

        maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
                                            [-1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 0, 1]], dtype=float)
        cam_extrinsics = cam_w_extrinsics @ maxtrix_camera2camera_w
        image_h, image_w, _ = image.shape
        cam_intrinsic = np.array(gt['intrinsic'])
        cam_extrinsics_persformer = copy.deepcopy(cam_w_extrinsics)
        R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        cam_extrinsics_persformer[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), cam_extrinsics_persformer[:3, :3]),
            R_vg), R_gc)
        cam_extrinsics_persformer[0:2, 3] = 0.0
        #  np.linalg.inv(maxtrix_camera2camera_w) : waymo camera to normal
        matrix_lane2persformer = cam_extrinsics_persformer @ np.linalg.inv(maxtrix_camera2camera_w)
        
        cam_intrinsic[0] *= self.input_w / image_w
        cam_intrinsic[1] *= self.input_h / image_h
        intrinsic = torch.tensor(cam_intrinsic)
        road2cam = np.linalg.inv(matrix_lane2persformer)
        road2cam = torch.tensor(road2cam)
        
        transformed = self.trans_image(image=image)
        image = transformed["image"]
        
        heightmap_gt = torch.tensor(bev_height_map_mask)
        
        return image, self.cnt_list[idx], intrinsic.float(), road2cam.float(), heightmap_gt.float()

    def __len__(self):
        return len(self.cnt_list)


if __name__ == "__main__":
    ''' parameter from config '''
    from utils.config_util import load_config_module
    config_file = '/mnt/ve_perception/wangruihao/code/BEV-LaneDet/tools/openlane_config.py'
    configs = load_config_module(config_file)
    dataset = configs.val_dataset()
    for item in dataset:
        continue