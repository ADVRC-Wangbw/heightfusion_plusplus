# Configuration

## Environment

Python 3.8, CUDA 11.1, and PyTorch 1.9.0 are recommended. Install the project dependencies and install this directory in development mode:

```bash
pip install -r requirements.txt
python setup.py develop
```

Use MMDetection3D `0.18.1` built for the installed PyTorch and CUDA versions.

## Dataset Layout

Set the dataset root with `data_root` in the experiment file. The dataset directory must contain images, annotations, calibration files, `ImageSets` splits, and the generated validation information file:

```text
DATA_ROOT/
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── label_2/
│   ├── velodyne/        # Required for LiDAR configurations
│   └── radar/           # Required for radar configurations
└── ImageSets/
    ├── train.txt
    └── val.txt
```

Generate the information file for the selected dataset layout:

```bash
python scripts/gen_info_heightfusion++.py --data_root DATA_ROOT
```

## Experiment Settings

Select or create an experiment file under `exps/`, then configure:

- `data_root`: dataset root directory.
- `radar_root`: radar dataset root; required only by radar configurations.
- `ida_aug_conf.final_dim`: input image size.
- `backbone_conf` and `head_conf`: camera backbone and detection-head settings.
- `voxelize_cfg` and `pts_middle_encoder`: LiDAR point-cloud range, voxel size, and encoder settings.
- `radar_voxelize_cfg` and `radar_middle_encoder`: radar point-cloud range and encoder settings.
- `batch_size_per_device`, `max_epochs`, `optimizer`, and `lr_scheduler`: training-resource and optimization settings.

The selected camera, LiDAR, radar, and fusion configuration must match the available data and its fields. Pass the experiment-file path as `EXP_PATH` when training or evaluating.
