import numpy as np
import torch
from scipy.interpolate import CubicSpline


def mean_col_by_row_with_offset_z(seg, offset_y, z):
    assert (len(seg.shape) == 2)

    center_ids = np.unique(seg[seg > 0])
    lines = []
    for idx, cid in enumerate(center_ids):  # 一个id
        cols, rows, z_val = [], [], []
        for y_op in range(seg.shape[0]):  # Every row
            condition = seg[y_op, :] == cid
            x_op = np.where(condition)[0]  # All pos in this row
            # print(x_op)
            z_op = z[y_op, :]
            offset_op = offset_y[y_op, :]
            if x_op.size > 0:
                offset_op = offset_op[x_op]
                z_op = np.mean(z_op[x_op])
                z_val.append(z_op)
                x_op_with_offset = x_op + offset_op
                x_op = np.mean(x_op_with_offset)  # mean pos
                cols.append(x_op)
                rows.append(y_op)
        lines.append((cols, rows, z_val))
    return lines


def bev_instance2points_with_offset_z(ids: np.ndarray, max_x=50, meter_per_pixal=(0.2, 0.2), offset_y=None, Z=None):
    center = ids.shape[1] / 2
    lines = mean_col_by_row_with_offset_z(ids, offset_y, Z)
    # print(lines)
    points = []
    # for i in range(1, ids.max()):

    for y, x, z in lines:  # cols, rows
        # x, y = np.where(ids == 1)
        x = np.array(x)[::-1]
        y = np.array(y)[::-1]
        z = np.array(z)[::-1]

        x = max_x / meter_per_pixal[0] - x
        y = y * meter_per_pixal[1]
        y -= center * meter_per_pixal[1]
        x = x * meter_per_pixal[0] 

        y *= -1.0  # Vector is from right to left
        if len(x) < 2:
            continue
        spline = CubicSpline(x, y, extrapolate=False)
        points.append((x, y, z, spline))
    return points



class PostProcessingModule(torch.nn.Module):
    # embedding 이 따로 필요하지 않은, loss 를 계산하기 위해 processing이 필요한 경우에 이 버전을 사용한다.
    def __init__(self, post_conf, post_emb_margin, post_min_cluster_size, max_x, meter_per_pixel, image_shape):
        super().__init__()
        self.post_conf = post_conf
        self.post_emb_margin = post_emb_margin
        self.post_min_cluster_size = post_min_cluster_size
        self.max_x = max_x
        self.meter_per_pixel = meter_per_pixel
        self.image_h, self.image_w = image_shape

    def forward(self, pred_, height_, intrinsic, extrinsic, projection=False):
        seg = pred_[0]  # [batch_size, 1, 200, 48]
        embedding = pred_[1] # [batch_size, num_channels, 200, 48]
        offset_y = torch.sigmoid(pred_[2])  # [batch_size, 1, 200, 48]
        z_pred = height_  # [batch_size, 1, 200, 48]
        b, nd, h, w = embedding.shape
        if nd > 1:
            c = self.collect_and_cluster_nd(seg, embedding, self.post_conf, self.post_emb_margin)

        else:
            ret = self.collect_embedding_with_position(seg, embedding, self.post_conf)
            c = self.naive_cluster(ret, self.post_emb_margin)
            
        lanes = torch.zeros_like(seg, dtype=torch.uint8,device='cuda:0')
        for b in range(len(c[0])):
            cids = c[0][b]
            centers = c[1][b]

            x_indices = torch.tensor([x for x, y, id in cids], device=lanes.device, dtype=torch.long)   
            y_indices = torch.tensor([y for x, y, id in cids], device=lanes.device, dtype=torch.long)
            ids = torch.tensor([id for x, y, id in cids], device=lanes.device, dtype=torch.long)

            # Extract cluster sizes and filter by minimum cluster size
            cluster_sizes = torch.tensor([centers[id][1] for id in ids], device=lanes.device, dtype=torch.long)
            valid_clusters = cluster_sizes >= self.post_min_cluster_size

            # Apply valid clusters to lanes
            lanes[b, 0, x_indices[valid_clusters], y_indices[valid_clusters]] = (ids[valid_clusters] + 1).to(torch.uint8)


        # batch_lines = self.mean_col_by_row_with_offset_z_batch(lanes, offset_y, z_pred)
        batch_points = self.mean_col_and_process_lanes(lanes, offset_y, z_pred, self.max_x, self.meter_per_pixel, w)
        if projection:
            batched_image = self.projection2d(batch_points, intrinsic, extrinsic)
            return batch_points, batched_image
        return batch_points
    
    def mean_col_and_process_lanes(self, seg, offset_y_batch, z_batch, max_x, meter_per_pixel, w):
        assert (len(seg.shape) == 4)
        assert (len(offset_y_batch.shape) == 4)
        batch_size = seg.shape[0]
        batch_points = []

        for b in range(batch_size):
            ids = seg[b, 0]
            offset_y = offset_y_batch[b, 0]
            z = z_batch[b, 0]

            # Extract unique IDs, ignoring zero (background)
            center_ids = torch.unique(ids[ids > 0])

            points = []
            for cid in center_ids:
                # Create masks and operations for the entire batch of `cid`
                mask = ids == cid
                y_indices, x_indices = torch.nonzero(mask, as_tuple=True)

                if y_indices.numel() == 0:
                    continue

                # Get offset and z values where mask is true
                offset_values = offset_y[mask]
                z_values = z[mask]

                # Adjust x coordinates and calculate mean values per unique y_index
                x_adjusted = x_indices.float() + offset_values
                # Calculate mean x, y, z per unique y_index
                unique_rows, inverse_indices = torch.unique_consecutive(y_indices, return_inverse=True)
                mean_x_per_row = torch.zeros_like(unique_rows, dtype=torch.float32)
                mean_z_per_row = torch.zeros_like(unique_rows, dtype=torch.float32)
                counts_per_row = torch.bincount(inverse_indices, minlength=len(unique_rows))

                mean_x_per_row.index_add_(0, inverse_indices, x_adjusted)
                mean_z_per_row.index_add_(0, inverse_indices, z_values)

                mean_x_per_row /= counts_per_row.float()
                mean_z_per_row /= counts_per_row.float()
                rows_adjusted = unique_rows.float() 

                # Reverse the tensors
                mean_x_per_row = mean_x_per_row.flip(dims=[0])  # y 
                rows_adjusted = rows_adjusted.flip(dims=[0]) # x
                mean_z_per_row = mean_z_per_row.flip(dims=[0])

                # Perform the calculations
                x = max_x / meter_per_pixel[0] - rows_adjusted
                y = mean_x_per_row * meter_per_pixel[1]
                y -= (w // 2) * meter_per_pixel[1]
                x = x * meter_per_pixel[0]

                # Invert y-axis
                y = -y

                # Check for sufficient points for spline
                if len(x) >= 2:
                    points.append((x, y, mean_z_per_row))

            batch_points.append(points)

        return batch_points

    def ego2image(self, ego_points, camera_intrinsic, ego2camera_matrix, output_2d =(144,256)):
        camera_points = torch.matmul(ego2camera_matrix[:3, :3], ego_points) + \
                    ego2camera_matrix[:3, 3].unsqueeze(1)
        image_points_ = torch.matmul(camera_intrinsic, camera_points)
        image_points = image_points_ / image_points_[2, :]
        mask = (image_points[0] >= 0) & (image_points[0] < output_2d[1]) & \
           (image_points[1] >= 0) & (image_points[1] < output_2d[0])
        image_points = image_points[:, mask]
        return image_points

    
    def projection2d(self, batch_lines, intrinsics, extrinsic, output_2d=(144, 256)):
        b = len(batch_lines)
        pred_2d = torch.zeros((b, output_2d[0], output_2d[1]), device='cuda:0')

        for batch_idx, lines in enumerate(batch_lines):
            intrinsic = intrinsics[batch_idx].clone()
            intrinsic[0] *= output_2d[1] / self.image_w
            intrinsic[1] *= output_2d[0] / self.image_h

            all_lanes = []
            for x, y, z in lines:
                lane = torch.stack([x, y, z], dim=0)
                all_lanes.append(lane)

            if not all_lanes:
                continue

            all_lanes = torch.cat(all_lanes, dim=1)
            image_points = self.ego2image(all_lanes, intrinsic, extrinsic[batch_idx])

            # Convert to integer pixel values
            x = image_points[0, :].long()
            y = image_points[1, :].long()

            # Set pixel values in the prediction tensor
            pred_2d[batch_idx, y, x] = 1

        return pred_2d

    def collect_and_cluster_nd(self, seg, embedding, conf, emb_margin):
        batch_centers = []
        batch_cids = []

        for b in range(seg.shape[0]):
            # 현재 배치의 세그먼트 데이터
            current_seg = seg[b, 0]  # shape: [height, width]

            # 현재 배치의 임베딩 데이터
            current_emb = embedding[b]  # shape: [num_features, height, width]

            # conf 이상의 값에 대한 마스크 생성
            mask = current_seg >= conf

            # 마스크에서 True인 위치의 인덱스를 추출
            i_indices, j_indices = torch.where(mask)

            if len(i_indices) == 0:
                batch_centers.append([])
                batch_cids.append([])
                continue

            # 해당 인덱스를 사용하여 현재 배치에서 임베딩 추출
            embeddings = current_emb[:, i_indices, j_indices].t()  # shape: [num_selected, num_features]

            # 초기화
            cids = torch.full((len(i_indices),), -1, dtype=torch.int64, device=embeddings.device)
            centers = []
            counts = []

            for idx in range(len(i_indices)):
                emb = embeddings[idx]
                if len(centers) == 0:
                    centers.append(emb)
                    counts.append(1)
                    cids[idx] = 0
                    continue

                # 모든 현재 센터들에 대한 거리 계산
                center_tensors = torch.stack(centers)
                distances = torch.norm(center_tensors - emb, dim=1)

                # 가장 가까운 센터 찾기
                min_dist, min_cid = torch.min(distances, dim=0)
                if min_dist < emb_margin:
                    # 센터와 카운트 업데이트
                    center_count = counts[min_cid]
                    new_center = (center_tensors[min_cid] * center_count + emb) / (center_count + 1)
                    centers[min_cid] = new_center
                    counts[min_cid] += 1
                    cids[idx] = min_cid.item()
                else:
                    centers.append(emb)
                    counts.append(1)
                    cids[idx] = len(centers) - 1

            batch_centers.append([(center, count) for center, count in zip(centers, counts)])
            batch_cids.append([(i.item(), j.item(), cid) for i, j, cid in zip(i_indices, j_indices, cids)])

        return batch_cids, batch_centers
    
    
