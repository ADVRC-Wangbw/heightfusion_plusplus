# Empty KITTI Dataset Example

This directory is an empty KITTI-style dataset skeleton for the project. It
contains no images, point clouds, radar scans, annotations, or calibration data.

```text
kitti_example/
|-- ImageSets/
|   |-- train.txt
|   |-- trainval.txt
|   |-- val.txt
|   `-- test.txt
|-- training/
|   |-- calib/
|   |-- image_2/
|   |-- label_2/
|   |-- radar/       # Optional; required by radar configurations.
|   `-- velodyne/
`-- README.md
```

Place files with matching frame IDs in the corresponding directories. For
example, `000001.bin`, `000001.txt`, and `000001.png` refer to the same frame.
Add one frame ID per line to the applicable `ImageSets` split files. The
`radar/` directory is project-specific and may be omitted for LiDAR-only runs.