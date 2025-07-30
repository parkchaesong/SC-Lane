# SC‑Lane

Official implementation of **SC‑Lane** (ICCV 2025)

---

### Overview

**SC‑Lane: Slope‑aware and Consistent Road Height Estimation Framework for 3D Lane Detection**

SC‑Lane proposes a *more refined* height‑estimation framework that models road slope and temporal consistency for accurate 3D lane detection. In addition, it introduces the **first height‑estimation benchmark** tailored specifically for lane‑detection tasks.

> **Code release:** The full source code, pretrained models, and benchmark toolkit will be **publicly released here before ICCV 2025**. Stay tuned!

---

## 0. Dataset Preparation

SC‑Lane shares the same data setup as HeightLane.

### Step 1 – Download OpenLane Dataset

Follow the instructions in the [OpenLane Dataset README](https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md) to download the full dataset.

After downloading, your directory should look like:

```
<root>/openlane/
├── images/
├── training/
└── validation/
```

### Step 2 – Download Height‑Map Data

Download the height‑map archive:
[https://147.46.111.77:1402/sharing/jplpr7ROl](https://147.46.111.77:1402/sharing/jplpr7ROl)

Unzip to obtain `Openlane_height/` containing:

```
Openlane_height/
├── heightmap_training/
└── heightmap_validation/
```

Move these folders into the `openlane/` directory so the final layout is:

```
<root>/openlane/
├── images/
├── training/
├── validation/
├── heightmap_training/
└── heightmap_validation/
```

---

### Acknowledgments

This repository builds upon

* [**BEV‑LaneDet**](https://github.com/gigo-team/bev_lane_det)
* [**HeightLane**](https://github.com/parkchaesong/HeightLane)

Many thanks to the authors of these projects for laying the groundwork.
