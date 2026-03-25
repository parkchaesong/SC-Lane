import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loader.bev_road.openlane_data import OpenLane_dataset_with_offset,OpenLane_dataset_with_offset_val
from models.model.sc_lane import SCLane

''' data split '''
train_gt_paths = '/media/vdcl/T9/openlane/training'
train_image_paths = '/media/vdcl/T9/openlane/images/training'
train_map_paths = '/media/vdcl/T9/openlane/heightmap_train'
val_gt_paths = '/media/vdcl/T9/openlane/validation'
val_image_paths = '/media/vdcl/T9/openlane/images/validation'
val_map_paths = '/media/vdcl/T9/openlane/heightmap_validation''

model_save_path = "./checkpoints/sc_lane"

input_shape = (600,800)
output_2d_shape = (144,256)

''' BEV range '''
x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5 # grid size
bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel),int((y_range[1] - y_range[0]) / meter_per_pixel))

loader_args = dict(
    batch_size=4,
    num_workers=4
)


''' model '''
def model():
    return SCLane(bev_shape=bev_shape, image_shape=input_shape, output_2d_shape=output_2d_shape,
                  train=True, use_img=False, x_range=x_range, y_range=y_range, meter_per_pixel=meter_per_pixel)

def val_model():
    return SCLane(bev_shape=bev_shape, image_shape=input_shape, output_2d_shape=output_2d_shape,
                  train=False, use_img=False, x_range=x_range, y_range=y_range, meter_per_pixel=meter_per_pixel)


''' optimizer '''
epochs = 50
optimizer = AdamW
optimizer_params = dict(
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
scheduler = CosineAnnealingLR



def train_dataset():
    train_trans = A.Compose([
                    A.Resize(height=input_shape[0], width=input_shape[1]),
                    A.MotionBlur(p=0.2),
                    A.RandomBrightnessContrast(),
                    A.ColorJitter(p=0.1),
                    A.Normalize(),
                    ToTensorV2()
                    ])
    train_data = OpenLane_dataset_with_offset(train_image_paths, train_gt_paths, train_map_paths,
                                              x_range, y_range, meter_per_pixel,
                                              train_trans, output_2d_shape, input_shape)

    return train_data


def val_dataset():
    trans_image = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2()])
    val_data = OpenLane_dataset_with_offset_val(val_image_paths, val_gt_paths, val_map_paths,
                                                trans_image, x_range, y_range, meter_per_pixel, input_shape)
    return val_data
