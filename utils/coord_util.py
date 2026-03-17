import numpy as np
import cv2


def ego2image(ego_points, camera_intrinsic, ego2camera_matrix):
    """
    :param ego_points:  3*n
    :param camera_intrinsic: 3*3
    :param camera2ego_matrix:  4*4
    :return:
    """
    camera_points = np.dot(ego2camera_matrix[:3, :3], ego_points) + \
                    ego2camera_matrix[:3, 3].reshape(3, 1)
    image_points_ = camera_intrinsic @ camera_points
    image_points = image_points_ / image_points_[2]
    return image_points


def ego2image_filtered(ego_points, camera_intrinsic, w, h, ego2camera_matrix, masking= False):
    """
    :param ego_points:  3*n
    :param camera_intrinsic: 3*3
    :param camera2ego_matrix:  4*4
    :return:
    """
    camera_points = np.dot(ego2camera_matrix[:3, :3], ego_points) + \
                    ego2camera_matrix[:3, 3].reshape(3, 1)
    image_points_ = camera_intrinsic @ camera_points
    image_points = image_points_ / image_points_[2]
    # Filter out points outside the image boundary
    mask = (image_points[0] >= 0) & (image_points[0] < w) & \
           (image_points[1] >= 0) & (image_points[1] < h)
    image_points = image_points[:, mask]
    return image_points

def IPM2ego_matrix(ipm_center=None, m_per_pixel=None, ipm_points=None, ego_points=None):
    if ipm_points is None:
        center_x, center_y = ipm_center[0] * m_per_pixel, ipm_center[1] * m_per_pixel
        M = np.array([[-m_per_pixel, 0, center_x], [0, -m_per_pixel, center_y]])
    else:
        pts1 = np.float32(ipm_points)
        pts2 = np.float32(ego_points)
        M = cv2.getAffineTransform(pts1, pts2)
    return M


if __name__ == '__main__':
    ego_points = np.array([[0, 0], [0, 12], [50, 12]])
    image_points = np.array([[250, 60], [250, 0], [0, 0]])
    print(IPM2ego_matrix(ipm_points=image_points, ego_points=ego_points))
