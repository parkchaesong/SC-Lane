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
            gt_list = os.listdir(gt_paths)
            for cnt in gt_list:
                self.cnt_list.append([card, cnt])

        self.sep_map_paths = "/media/vdcl/T9/Waymo/map_data_training_seperate"
        ''' transform loader '''
        self.output2d_size = output_2d_shape
        self.trans_image = data_trans
        self.input_h, self.input_w = input_shape
        self.ipm_h, self.ipm_w = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int(
            (self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)
        
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
    
    def project_point_cloud_to_image_plane(self, point_cloud, extrinsic, intrinsic):
        # Homogeneous coordinates for 3D point cloud
        point_cloud_homo = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

        # Project 3D points to image plane
        image_points_homogeneous = intrinsic @ extrinsic[:3,:] @ point_cloud_homo.T

        # Convert homogeneous coordinates to Cartesian coordinates
        image_points_cartesian = image_points_homogeneous[:2] / image_points_homogeneous[2]

        return image_points_cartesian
    
    def extract_heightmap_within_range(self, pc, vehicle_pose_matrix, cam_w_extrinsics, forward_range, lateral_range):
        gt_pc = np.concatenate(pc, axis=1)

        # Extract rotation matrix and translation vector from vehicle pose matrix
        R = vehicle_pose_matrix[:3, :3]  # Rotation matrix
        T = vehicle_pose_matrix[:3, 3]   # Translation vector

        R_inverse = np.linalg.inv(R)
        # Translate point cloud to vehicle position
        translated_point_cloud = gt_pc.T[:,:3] - T

        # Rotate point cloud to vehicle coordinate system
        point_cloud_vehicle = (R_inverse@translated_point_cloud.T).T
        
        # Extract rotation matrix and translation vector from vehicle pose matrix
        R_ex = cam_w_extrinsics[:3, :3]  # Rotation matrix
        T_ex = cam_w_extrinsics[:3, 3]   # Translation vector
        
        R_inverse_ex = np.linalg.inv(R_ex)
        # Translate point cloud to vehicle position
        translated_point_cloud_ex =point_cloud_vehicle[:,:3] - T_ex
        
        # Rotate point cloud to camera coordinate system
        point_cloud_camera = (R_inverse_ex@translated_point_cloud_ex.T).T

        x = point_cloud_camera[:, 0]
        y = point_cloud_camera[:, 1]

        # Filter points within the specified range
        mask = (x >= forward_range[0]) & (x <= forward_range[1]) & (y >=lateral_range[0]) & (y <= lateral_range[1])
        filtered_point_cloud = point_cloud_camera[mask]
        return filtered_point_cloud
        
    def generate_bev_height_map(self, point_cloud, resolution=(200,48), meter_per_pixel=0.5, vehicle_height=0):
        bev_resolution_x, bev_resolution_y = resolution
            
        # Initialize BEV height map
        bev_height_map = np.zeros((bev_resolution_x, bev_resolution_y))
        x_grid, y_grid = np.meshgrid(np.arange(bev_resolution_x), np.arange(bev_resolution_y), indexing='ij')
        # 각 픽셀에 대한 Z 값의 합과 개수를 저장할 배열 초기화
        z_sum_array = np.zeros((bev_resolution_x, bev_resolution_y))
        count_array = np.zeros((bev_resolution_x, bev_resolution_y))
        
        # 포인트 클라우드 반복하면서 각 픽셀에 대한 Z 값의 합과 개수 계산
        x_coords = point_cloud[:, 0].astype(int)
        y_coords = point_cloud[:, 1].astype(int)
        z_values = point_cloud[:, 2]
        
        # 유효한 픽셀 좌표 찾기
        valid_pixels_mask = (x_coords >= 0) & (x_coords < bev_resolution_x) & (y_coords >= 0) & (y_coords < bev_resolution_y)

        # 각 픽셀에 대한 Z 값의 합과 개수 계산
        np.add.at(z_sum_array, (x_coords[valid_pixels_mask], y_coords[valid_pixels_mask]), z_values[valid_pixels_mask])
        np.add.at(count_array, (x_coords[valid_pixels_mask], y_coords[valid_pixels_mask]), 1)

        # 계산된 Z 값의 합을 개수로 나누어 평균 계산
        with np.errstate(divide='ignore', invalid='ignore'):  # 0으로 나누는 오류 무시
            bev_height_map = np.divide(z_sum_array, count_array, out=np.zeros_like(z_sum_array), where=count_array != 0)
            
        if np.sum(valid_pixels_mask) >= 96:
            known_points = np.array((x_coords[valid_pixels_mask], y_coords[valid_pixels_mask])).T
            known_values = bev_height_map[known_points[:, 0], known_points[:, 1]]
            interpolated_values = griddata(known_points, known_values, (x_grid, y_grid), method='linear')
    
            # NaN 값을 0으로 채우기 (선택사항)
            interpolated_map = np.nan_to_num(interpolated_values, nan=0.0)
        else:
        # 유효한 포인트가 없는 경우 기본값을 사용 (예: 모두 0으로 설정)
            interpolated_map = bev_height_map
    
        #distances = (199 - np.arange(bev_resolution_x)) / 2 + 3
        #height_angle_map = interpolated_map / distances.reshape(-1, 1)
        #height_angle_map = np.arctan(height_angle_map)
        # print(distances)
        # print(interpolated_map)
        # print(np.unique(interpolated_map))
        return interpolated_map
    
    def generate_binary_mask(self, bev_height_map):
        # Initialize binary mask
        binary_mask = np.zeros_like(bev_height_map, dtype=np.int)

        # Mark pixels with points as 1
        binary_mask[bev_height_map != 0] = 1

        return binary_mask
    
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
                # if offset_y > 1:
                #     offset_y = 1
                # if offset_y < 0:
                #     offset_y = 0
                offset_map[row][col] = np.clip(offset_y, 0, 1)
                z_map[row][col] = z

        return ipm_image, offset_map, z_map

    def get_seg_offset(self, idx, smooth=False):
        
        gt_path = os.path.join(self.gt_paths, self.cnt_list[idx][0], self.cnt_list[idx][1])
        # print(self.cnt_list[idx][0])
        map_path = os.path.join(self.map_paths, self.cnt_list[idx][0]+'.tfrecord_dbinfos.pkl')
        seperate_map_path = os.path.join(self.sep_map_paths, self.cnt_list[idx][0], self.cnt_list[idx][1].replace('json', 'pkl'))
        # print(map_path, seperate_map_path)
        image_path = os.path.join(self.image_paths, self.cnt_list[idx][0], self.cnt_list[idx][1].replace('json', 'jpg'))
        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        # cam_w_extrinsics : Waymo camera coordinate to Waymovehicle coordinate 
        cam_w_extrinsics = np.array(gt['extrinsic'])
        maxtrix_camera2camera_w = np.array([[0, 0, 1, 0],
                                            [-1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 0, 1]], dtype=float)
        # cam_extrinsics : Waymo camera normal coordinate to Waymo vehicle coordinate
        cam_extrinsics = cam_w_extrinsics @ maxtrix_camera2camera_w  #
        # cam_extrinsics : Waymo vehicle coordinate to Waymo camera normal coordinate
        cam_extrinsics = np.linalg.inv(cam_extrinsics)
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
        #  np.linalg.inv(maxtrix_camera2camera_w) : waymo camera to normal
        matrix_lane2persformer = cam_extrinsics_persformer @ np.linalg.inv(maxtrix_camera2camera_w)

        cam_intrinsic = np.array(gt['intrinsic'])
        matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel), int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)
        
        with open(seperate_map_path,'rb') as map_f:
            map_pointcloud = pickle.load(map_f)
            
        vehicle_pose = np.array(gt['pose'])
        #map_pointcloud = map_db_infos["global_pointcloud"]
        # print(map_pointcloud)
        # global heightmap to local vehicle camera coordinate
        #roi_pointcloud = self.extract_heightmap_within_range(map_pointcloud, vehicle_pose,cam_w_extrinsics, self.x_range, self.y_range)
        
        roi_pointcloud = np.vstack((map_pointcloud.T, np.ones((1, map_pointcloud.shape[0]))))
        new_roi = matrix_lane2persformer @ roi_pointcloud
        roi = np.array([new_roi[1], -new_roi[0], new_roi[2]])
        ipm_roi = self.bev2ipm(roi, matrix_IPM2ego)
        
        bev_heightmap = self.generate_bev_height_map(ipm_roi.T, resolution=(self.ipm_h, self.ipm_w), meter_per_pixel=self.meter_per_pixel)
        bev_heightmask = self.generate_binary_mask(bev_heightmap)
        bev_height_map_mask = np.stack((bev_heightmap, bev_heightmask), axis=0)
        
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

            ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (ego_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)
            res_points_d[idx + 1] = res_points
        ipm_gt, offset_y_map, z_map = self.get_y_offset_and_z(res_points_d)
        # print(image.shape)
        return image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, matrix_lane2persformer, bev_height_map_mask

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        image, image_gt, ipm_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic, matrix_lane2persformer, bev_height_map_mask = self.get_seg_offset(idx)
        # print("z",np.unique(z_map))
        # print("height",np.unique(bev_height_map_mask[:,:,0]))
        image_h, image_w, _ = image.shape
        cam_intrinsic[0] *= self.input_w / image_w
        cam_intrinsic[1] *= self.input_h / image_h
        transformed = self.trans_image(image=image)
        road2cam = np.linalg.inv(matrix_lane2persformer)
        image = transformed["image"]
        intrinsic = torch.tensor(cam_intrinsic)
        extrinsic = torch.tensor(cam_extrinsics)
        road2cam = torch.tensor(road2cam)
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

        return image, ipm_gt_segment.float(), ipm_gt_instance.float(), ipm_gt_offset.float(), ipm_gt_z.float(), image_gt_segment.float(), image_gt_instance.float(), intrinsic.float(), extrinsic.float(), road2cam.float(), image_gt_heightmap.float()

    def __len__(self):
        return len(self.cnt_list)


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
    
    def extract_heightmap_within_range(self, pc, vehicle_pose_matrix, cam_w_extrinsics, forward_range, lateral_range):
        gt_pc = np.concatenate(pc, axis=1)

        # Extract rotation matrix and translation vector from vehicle pose matrix
        R = vehicle_pose_matrix[:3, :3]  # Rotation matrix
        T = vehicle_pose_matrix[:3, 3]   # Translation vector

        R_inverse = np.linalg.inv(R)
        # Translate point cloud to vehicle position
        translated_point_cloud = gt_pc.T[:,:3] - T

        # Rotate point cloud to vehicle coordinate system
        point_cloud_vehicle = (R_inverse@translated_point_cloud.T).T
        
        # Extract rotation matrix and translation vector from vehicle pose matrix
        R_ex = cam_w_extrinsics[:3, :3]  # Rotation matrix
        T_ex = cam_w_extrinsics[:3, 3]   # Translation vector
        
        R_inverse_ex = np.linalg.inv(R_ex)
        # Translate point cloud to vehicle position
        translated_point_cloud_ex =point_cloud_vehicle[:,:3] - T_ex
        
        # Rotate point cloud to camera coordinate system
        point_cloud_camera = (R_inverse_ex@translated_point_cloud_ex.T).T

        x = point_cloud_camera[:, 0]
        y = point_cloud_camera[:, 1]

        # Filter points within the specified range
        mask = (x >= forward_range[0]) & (x <= forward_range[1]) & (y >=lateral_range[0]) & (y <= lateral_range[1])
        filtered_point_cloud = point_cloud_camera[mask]
        return filtered_point_cloud
    
    def generate_bev_height_map(self, point_cloud, resolution=(200,48), meter_per_pixel=0.5, vehicle_height=0):
        bev_resolution_x, bev_resolution_y = resolution
            
        # Initialize BEV height map
        bev_height_map = np.zeros((bev_resolution_x, bev_resolution_y))
        x_grid, y_grid = np.meshgrid(np.arange(bev_resolution_x), np.arange(bev_resolution_y), indexing='ij')
        # 각 픽셀에 대한 Z 값의 합과 개수를 저장할 배열 초기화
        z_sum_array = np.zeros((bev_resolution_x, bev_resolution_y))
        count_array = np.zeros((bev_resolution_x, bev_resolution_y))
        
        # 포인트 클라우드 반복하면서 각 픽셀에 대한 Z 값의 합과 개수 계산
        x_coords = point_cloud[:, 0].astype(int)
        y_coords = point_cloud[:, 1].astype(int)
        z_values = point_cloud[:, 2]
        
        # 유효한 픽셀 좌표 찾기
        valid_pixels_mask = (x_coords >= 0) & (x_coords < bev_resolution_x) & (y_coords >= 0) & (y_coords < bev_resolution_y)

        # 각 픽셀에 대한 Z 값의 합과 개수 계산
        np.add.at(z_sum_array, (x_coords[valid_pixels_mask], y_coords[valid_pixels_mask]), z_values[valid_pixels_mask])
        np.add.at(count_array, (x_coords[valid_pixels_mask], y_coords[valid_pixels_mask]), 1)

        # 계산된 Z 값의 합을 개수로 나누어 평균 계산
        with np.errstate(divide='ignore', invalid='ignore'):  # 0으로 나누는 오류 무시
            bev_height_map = np.divide(z_sum_array, count_array, out=np.zeros_like(z_sum_array), where=count_array != 0)
            
        if np.sum(valid_pixels_mask) >= 96:
            known_points = np.array((x_coords[valid_pixels_mask], y_coords[valid_pixels_mask])).T
            known_values = bev_height_map[known_points[:, 0], known_points[:, 1]]
            interpolated_values = griddata(known_points, known_values, (x_grid, y_grid), method='linear')
    
            # NaN 값을 0으로 채우기 (선택사항)
            interpolated_map = np.nan_to_num(interpolated_values, nan=0.0)
        else:
        # 유효한 포인트가 없는 경우 기본값을 사용 (예: 모두 0으로 설정)
            interpolated_map = bev_height_map
    
        #distances = (199 - np.arange(bev_resolution_x)) / 2 + 3
        #height_angle_map = interpolated_map / distances.reshape(-1, 1)
        #height_angle_map = np.arctan(height_angle_map)
        # print(distances)
        # print(interpolated_map)
        # print(np.unique(interpolated_map))
        return interpolated_map
    
    def __getitem__(self, idx):
        matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel), int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=0.5)
        '''get image '''
        gt_path = os.path.join(self.gt_paths, self.cnt_list[idx][0], self.cnt_list[idx][1])
        image_path = os.path.join(self.image_paths, self.cnt_list[idx][0], self.cnt_list[idx][1].replace('json', 'jpg'))
        map_path = os.path.join(self.map_paths, self.cnt_list[idx][0]+'.tfrecord_dbinfos.pkl')
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
        
        '''
        with open(map_path,'rb') as map_f:
            map_db_infos = pickle.load(map_f)
        vehicle_pose = np.array(gt['pose'])
        map_pointcloud = map_db_infos["global_pointcloud"]
        # print(map_pointcloud)
        # global heightmap to local vehicle camera coordinate
        roi_pointcloud = self.extract_heightmap_within_range(map_pointcloud, vehicle_pose,cam_w_extrinsics, self.x_range, self.y_range)
        
        
        roi_pointcloud = np.vstack((roi_pointcloud.T, np.ones((1, roi_pointcloud.shape[0])))) 
        new_roi = matrix_lane2persformer @ roi_pointcloud # road1 coordinate [-횡, 종, z]
        roi = np.array([new_roi[1], -new_roi[0], new_roi[2]]) # road2 coordinate [종, 횡, z]
        ipm_roi = self.bev2ipm(roi, matrix_IPM2ego) # 3d point to bev coordinate (grid)
        # pdb.set_trace()
        gt_bev_heightmap = self.generate_bev_height_map(ipm_roi.T, resolution=(200, 48), meter_per_pixel=0.5) '''
        
        '''
        [[ 0.99989268 -0.00599321  0.01336787  1.53891424]
         [ 0.00604224  0.99997516 -0.00363024 -0.02363394]
         [-0.01334578  0.00371062  0.99990406  2.11527057]
         [ 0.          0.          0.          1.        ]] cam extrinsic

         [ x: 0.0036306, y: 0.0133683, z: 0.0059938 ]

        '''

        '''
        [[ 0.99989268, -0.00599321,  0.01336787], [ 0.00604224,  0.99997516, -0.00363024], [-0.01334578 , 0.00371062 , 0.99990406]]] cam extrinsic

        '''

        
        cam_intrinsic[0] *= self.input_w / image_w
        cam_intrinsic[1] *= self.input_h / image_h
        intrinsic = torch.tensor(cam_intrinsic)
        road2cam = np.linalg.inv(matrix_lane2persformer)
        cam2road = torch.tensor(road2cam)
        heightmap = torch.tensor(matrix_lane2persformer).unsqueeze(0)

        transformed = self.trans_image(image=image)
        image = transformed["image"]
        return image, self.cnt_list[idx], intrinsic.float(), cam2road.float(), heightmap.float()

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