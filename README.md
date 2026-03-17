# SC-Lane

Official implementation of **SC-Lane** (ICCV 2025)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://parkchaesong.github.io/sclane/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2508.10411)

> **Last updated:** 2026-03-17

---

### Overview

**SC-Lane: Slope-aware and Consistent Road Height Estimation Framework for 3D Lane Detection**

SC-Lane proposes a *more refined* height-estimation framework that models road slope and temporal consistency for accurate 3D lane detection. In addition, it introduces the **first height-estimation benchmark** tailored specifically for lane-detection tasks.

---

### 0. Dataset Preparation

#### Step 1: Download OpenLane Dataset

Follow the instructions from the [OpenLane Dataset README](https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md) to download the full dataset.

After downloading, your directory structure should look like:

```
<root>/openlane/
├── images/
├── training/
└── validation/
```

#### Step 2: Download Height Map Data

Download the height map data from the following link:
[https://147.46.111.77:1402/sharing/jplpr7ROl](https://147.46.111.77:1402/sharing/jplpr7ROl)

Unzip the archive to get a folder named `Openlane_height`. Inside it, you will find folders such as:

```
Openlane_height/
├── heightmap_training/
└── heightmap_validation/
```

Move these two folders into the previously created `openlane/` folder so that the final structure is:

```
<root>/openlane/
├── images/
├── training/
├── validation/
├── heightmap_training/
└── heightmap_validation/
```

#### Step 3: Update Configuration

Set the paths to your `<root>/openlane` directory in `tools/sc_lane_config.py`:

```python
train_gt_paths = '/path/to/openlane/training'
train_image_paths = '/path/to/openlane/images/training'
train_map_paths = '/path/to/Waymo/map_data_training'
val_gt_paths = '/path/to/openlane/validation'
val_image_paths = '/path/to/openlane/images/validation'
val_map_paths = '/path/to/Waymo/map_data_validation'
```

---

### Directory Structure

To ensure everything works correctly, clone the repositories under a common parent directory like this:

```
<your_workspace>/
├── Deformable-DETR/
└── SC-Lane/
```

---

### Installation

#### 1. Clone this repository:

```bash
git clone https://github.com/parkchaesong/SC-Lane.git
```

#### 2. Clone the required dependency (Deformable-DETR) **in the same parent directory**:

```bash
git clone https://github.com/fundamentalvision/Deformable-DETR.git
```

#### 3. Compile CUDA operators:

Navigate to the operators directory and compile the necessary CUDA operators:

```bash
cd Deformable-DETR/models/ops
sh ./make.sh
```

#### 4. Install Python dependencies:

Return to the SC-Lane root directory and install the required packages:

```bash
cd ../../../SC-Lane
pip install -r requirement.txt
```

---

### Usage

#### Pretrained Checkpoint

Download the pretrained model from the following link:
[https://drive.google.com/file/d/1UwiKDp8WzGMRd_cLYOdCs8jiuqKQ6i4Z](https://drive.google.com/file/d/1UwiKDp8WzGMRd_cLYOdCs8jiuqKQ6i4Z/view?usp=sharing)

Place the downloaded file in the project root directory:

```
SC-Lane/
└── ckpt.pth
```

#### Validation

Once the dataset and pretrained checkpoint are ready, run:

```bash
python tools/val.py
```

This will evaluate the model on the OpenLane validation set using the pretrained checkpoint.

To also log results to Weights & Biases:

```bash
python tools/val.py --wandb
```

#### Training

To train SC-Lane from scratch:

```bash
python tools/train.py
```

Training config (learning rate, epochs, batch size, etc.) is defined in `tools/sc_lane_config.py`. Checkpoints are saved to the path specified by `model_save_path` in the config (default: `./checkpoints/sc_lane/`).

---

### Citation

If you use SC-Lane in your research, please cite:

```
@inproceedings{park2025sc,
    title     = {SC-Lane: Slope-aware and Consistent Road Height Estimation Framework for 3D Lane Detection},
    author    = {Park, Chaesong and Seo, Eunbin and Hwang, Jihyeon and Lim, Jongwoo},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages     = {28407--28416},
    year      = {2025}
}
```

---

### Acknowledgments

This repository builds upon

* [**BEV-LaneDet**](https://github.com/gigo-team/bev_lane_det)
* [**HeightLane**](https://github.com/parkchaesong/HeightLane)

Many thanks to the authors of these projects for laying the groundwork.
