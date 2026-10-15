# HeightFusion++ result refinement

`refine_results.py` is a standalone temporal post-processing tool for 3D
detections. It accepts one pickle file and writes one pickle file. It does not
load a model, point cloud, calibration, dataset configuration, or framework.

Run it with:

```bash
python refine_results.py --input results.pkl --output refined_results.pkl
```

The input must preserve chronological frame order and use one of these forms:

```python
{"frames": [{"frame_id": "000001", "detections": [...]}, ...]}
{"results": {"000001": [...], "000002": [...]}}
[{"frame_id": "000001", "detections": [...]}, ...]
```

Each detection needs a class field (`label`, `detection_name`, or `name`), a
score field (`score` or `detection_score`), a three-value center field
(`translation` or `center`), `size = [length, width, height]`, and either
`yaw` or a quaternion `rotation = [w, x, y, z]`.

The tool only associates detections of the same class in adjacent frames. It
then applies score-weighted temporal smoothing to centers, sizes, and yaw for
tracks with at least two detections. All other keys and fields are preserved.

The `stages/` directory contains the three-stage MS3D workflow adapted from
[darrenjkt/MS3D](https://github.com/darrenjkt/MS3D), released under Apache-2.0.
Those scripts retain their upstream OpenPCDet dependency because KBF,
point-aware tracking, and static-object propagation require it. They use only
paths supplied through `--ps_cfg`; no absolute path is embedded in the scripts.
