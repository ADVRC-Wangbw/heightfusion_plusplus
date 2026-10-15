#!/usr/bin/env python3
"""Infer the highest-epoch BEVHeight++/HeightFusion checkpoint and score it.

All model/data paths are CLI inputs or inferred from checkpoint metadata.
Metrics reuse evaluate_nuscenes_style.py and the reference MMDetection3D
protocol: center-distance AP and a rescaled, three-TP-metric NDS.
Only evaluate trusted local Lightning checkpoints (torch.load uses pickle).
"""

import argparse
from copy import deepcopy
import csv
import gc
import importlib.util
from pathlib import Path
import re
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT / "pyramid_training"
EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_CLASSES = ("Car", "Pedestrian", "Cyclist")
DEFAULT_DISTANCE_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_PC_RANGE = (0.0, -51.2, -5.0, 102.4, 51.2, 3.0)
METRIC_NAMES = ("mAP", "NDS", "mATE", "mASE", "mAOE")
EPOCH_PATTERN = re.compile(r"(?:^|[^a-zA-Z])epoch[=_-]?(\d+)(?=$|[^\d])")
metric = None


def load_metric_module(metric_script):
    """Load the companion nuScenes-style metric implementation by file path."""
    metric_script = metric_script.expanduser().resolve()
    if not metric_script.is_file():
        raise FileNotFoundError(
            f"Metric implementation not found: {metric_script}. "
            "Pass --metric-script /path/to/evaluate_nuscenes_style.py.")
    spec = importlib.util.spec_from_file_location("evaluate_nuscenes_style", metric_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metric implementation: {metric_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-script", type=Path,
                        default=EVALUATION_ROOT / "evaluate_nuscenes_style.py",
                        help="Path to the companion nuScenes-style metric implementation.")
    parser.add_argument("--model-dir", type=Path, nargs="+", required=True,
                        help="One or more training directories containing checkpoints/*.ckpt.")
    parser.add_argument("--data-root", type=Path,
                        help="Dataset root; default uses checkpoint data_root or the same-name weather subset.")
    parser.add_argument("--radar-root", type=Path,
                        help="Optional Radar root override for C+R, L+R, C+L+R checkpoints.")
    parser.add_argument("--datasets-dir", type=Path, default=PROJECT_ROOT / "data/weather",
                        help="Parent of same-name weather subsets when relocating model directories.")
    parser.add_argument("--info-path", type=Path,
                        help="Validation PKL; default DATA_ROOT/DATASET_NAME_12hz_infos_val.pkl.")
    parser.add_argument("--experiment", type=Path,
                        help="Optional experiment .py override; normally inferred from checkpoint structure.")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional explicit checkpoint; otherwise choose highest saved epoch, including last.ckpt.")
    parser.add_argument("--prediction-json", type=Path,
                        help="Score existing/refined nuScenes predictions instead of rerunning inference; requires --output-dir.")
    parser.add_argument("--output-dir", type=Path,
                        help="Default MODEL_DIR/nuscenes_latest_eval; batch runs use subdirectories.")
    parser.add_argument("--gpus", type=int, choices=(1,), default=1,
                        help="Single-GPU inference. Select the physical GPU with CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-thr", type=float, default=1e-3)
    parser.add_argument("--nms-pre", type=int, default=1000)
    parser.add_argument("--max-num", type=int, default=500)
    parser.add_argument("--classes", nargs="+", choices=DEFAULT_CLASSES,
                        default=list(DEFAULT_CLASSES))
    parser.add_argument("--pc-range", nargs=6, type=float,
                        help="Override BEV evaluation range; default inferred from checkpoint head config.")
    parser.add_argument("--distance-thresholds", nargs="+", type=float,
                        default=list(DEFAULT_DISTANCE_THRESHOLDS))
    parser.add_argument("--tp-distance-threshold", type=float, default=2.0)
    parser.add_argument("--min-recall", type=float, default=0.1)
    parser.add_argument("--min-precision", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show checkpoint/data/config selection without GPU inference or output writes.")
    args = parser.parse_args(argv)
    if len(args.model_dir) > 1 and any(
        value is not None for value in (args.data_root, args.radar_root, args.info_path, args.checkpoint, args.prediction_json)
    ):
        parser.error("Dataset, checkpoint and prediction overrides require a single --model-dir")
    if args.prediction_json is not None and args.output_dir is None:
        parser.error("--prediction-json requires --output-dir to preserve original inference metrics")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if len(set(args.classes)) != len(args.classes):
        parser.error("--classes must not contain duplicates")
    return args


def load_checkpoint_metadata(path):
    """Read epoch/hparams on CPU, then release large weights/optimizer tensors."""
    import torch

    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Not a Lightning model checkpoint: {path}")
    epoch = checkpoint.get("epoch")
    if epoch is None or int(epoch) < 0:
        raise ValueError(f"Checkpoint has no valid epoch: {path}")
    hparams = checkpoint.get("hyper_parameters", {})
    state = checkpoint["state_dict"]
    result = {
        "epoch": int(epoch),
        "global_step": int(checkpoint.get("global_step", -1)),
        "hyper_parameters": deepcopy(dict(hparams)),
        "is_fusion": any(key.startswith(("model.pts_middle_encoder.", "model.lidar_encoder.",
                                         "model.radar_encoder.")) for key in state),
    }
    del state, checkpoint
    gc.collect()
    return result


def select_checkpoint(model_dir, checkpoint=None):
    """Sort epoch numbers numerically; inspect last/non-epoch names explicitly.

    Load metadata only for highest-numbered files and unnumbered files, not
    every multi-GB checkpoint. Stored epoch/step wins among these candidates.
    A numbered filename is preferred over last.ckpt when epoch/step both tie.
    """
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint, load_checkpoint_metadata(checkpoint)
    folder = model_dir / "checkpoints"
    files = sorted(path for path in folder.glob("*.ckpt") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No checkpoints/*.ckpt found in {model_dir}")
    numbered, unnumbered = [], []
    for path in files:
        match = EPOCH_PATTERN.search(path.stem)
        if match:
            numbered.append((int(match.group(1)), path))
        else:
            unnumbered.append(path)
    max_named_epoch = max((epoch for epoch, _ in numbered), default=-1)
    candidates = [path for epoch, path in numbered if epoch == max_named_epoch]
    candidates += unnumbered
    ranked = []
    for path in candidates:
        metadata = load_checkpoint_metadata(path)
        rank = (metadata["epoch"], metadata["global_step"],
                bool(EPOCH_PATTERN.search(path.stem)), path.stat().st_mtime_ns, path.name)
        ranked.append((rank, path, metadata))
    _, selected, metadata = max(ranked, key=lambda item: item[0])
    return selected, metadata


def resolve_data_root(args, model_dir, metadata):
    if args.data_root is not None:
        return args.data_root.expanduser().resolve()
    stored = metadata["hyper_parameters"].get("data_root")
    candidates = []
    if stored:
        path = Path(stored).expanduser()
        candidates.append(path if path.is_absolute() else PROJECT_ROOT / path)
    candidates.append(args.datasets_dir.expanduser().resolve() / model_dir.name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate dataset; supply --data-root or --datasets-dir")


def resolve_experiment(args, data_root, metadata):
    if args.experiment is not None:
        return args.experiment.expanduser().resolve()
    stem = "bev_height_plus_lss_r101_384_1280_256x256"
    hparams = metadata["hyper_parameters"]
    suffix = "geometry_v2" if hparams.get("image_geometry_version") == "resize_v2" else "retrain"
    if hparams.get("experiment_path"):
        path = Path(hparams["experiment_path"]).expanduser()
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if path.is_file():
            return path.resolve()
    if hparams.get("fusion_mode") in ("cr", "lr", "clr"):
        return PROJECT_ROOT / "exps/kitti" / f"{stem}_{hparams['fusion_mode']}_{data_root.name}_{suffix}.py"
    if metadata["is_fusion"]:
        stem += "_lidar"
        # Old fusion checkpoints contain seven tasks; do not truncate their
        # trained head by importing the corrected six-task weather wrapper.
        if len(hparams.get("head_conf", {}).get("tasks", [])) == 7:
            return PROJECT_ROOT / "exps/kitti" / f"{stem}.py"
    weather_config = PROJECT_ROOT / "exps/kitti" / f"{stem}_{data_root.name}_{suffix}.py"
    if weather_config.is_file():
        return weather_config
    return PROJECT_ROOT / "exps/kitti" / f"{stem}.py"


def resolve_run(args, model_dir, batch=False):
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    checkpoint, metadata = select_checkpoint(model_dir, args.checkpoint)
    data_root = resolve_data_root(args, model_dir, metadata)
    hparams = metadata["hyper_parameters"]
    suffix = "heightfusion_infos_val.pkl" if hparams.get("fusion_mode") in ("cr", "lr", "clr") else "12hz_infos_val.pkl"
    info_path = args.info_path or data_root / f"{data_root.name}_{suffix}"
    experiment = resolve_experiment(args, data_root, metadata)
    pc_range = args.pc_range
    if pc_range is None:
        pc_range = hparams.get("head_conf", {}).get("train_cfg", {}).get("point_cloud_range")
        if pc_range is None:
            pc_range = DEFAULT_PC_RANGE
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = model_dir / "nuscenes_latest_eval"
    elif batch:
        output_dir = output_dir / model_dir.parent.name / model_dir.name
    run = argparse.Namespace(**vars(args))
    run.model_dir = model_dir
    run.checkpoint = checkpoint
    run.data_root = data_root
    run.info_path = info_path.expanduser().resolve()
    run.experiment = experiment
    run.output_dir = output_dir.expanduser().resolve()
    run.inference_output_dir = run.output_dir / f"eval_{checkpoint.stem}"
    run.prediction_json = None
    run.pc_range = list(pc_range)
    run.model_init_overrides = {}
    if hparams.get("fusion_mode") in ("cr", "lr", "clr"):
        stored_radar = hparams.get("radar_root")
        radar_root = getattr(args, "radar_root", None)
        if radar_root is None and stored_radar:
            stored_radar = Path(stored_radar).expanduser()
            if not stored_radar.is_absolute():
                stored_radar = PROJECT_ROOT / stored_radar
            if stored_radar.is_dir():
                radar_root = stored_radar
        if radar_root is None:
            radar_root = data_root.parent / data_root.name.replace("_lidar_v10", "_radar_v9")
        radar_root = radar_root.expanduser().resolve()
        if not radar_root.is_dir():
            raise FileNotFoundError(f"Radar root not found: {radar_root}; supply --radar-root")
        run.model_init_overrides["radar_root"] = str(radar_root)
    if metadata["is_fusion"]:
        voxel_cfg = deepcopy(hparams.get("voxelize_cfg", {}))
        if voxel_cfg:
            # Historical fusion construction pops this config field before
            # saving hyperparameters. The original architecture uses True.
            voxel_cfg.setdefault("voxelize_reduce", True)
            run.model_init_overrides["voxelize_cfg"] = voxel_cfg
    metric.validate_args(run)
    if getattr(args, "prediction_json", None) is not None:
        # Here the checkpoint is provenance only, not a request to infer again.
        # Keep the standalone metric's incompatible inference/JSON flags intact.
        run.prediction_json = args.prediction_json.expanduser().resolve()
        if not run.prediction_json.is_file():
            raise FileNotFoundError(f"Prediction JSON not found: {run.prediction_json}")
    return run, metadata


def evaluate_run(run, metadata):
    # Validate GT before spending GPU time; never score a partial validation set.
    classes = tuple(run.classes)
    gt_boxes, sample_tokens, ego_poses, gt_counts, ignored_gt = metric.load_ground_truth(
        run.info_path, classes, run.pc_range)
    if not sample_tokens:
        raise ValueError(f"Empty validation dataset: {run.info_path}")
    provided_prediction = getattr(run, "prediction_json", None)
    prediction_path = (Path(provided_prediction) if provided_prediction is not None
                       else metric.run_checkpoint_inference(run))
    pred_boxes, pred_counts, ignored_pred, missing = metric.load_predictions(
        prediction_path, sample_tokens, ego_poses, classes, run.pc_range,
        run.score_thr, run.max_num)
    if missing:
        raise RuntimeError(f"Inference omitted {len(missing)} validation frames; refusing partial metrics")
    metrics, detailed = metric.evaluate(gt_boxes, pred_boxes, classes, run)
    overall = {name: metrics[f"Overall/{name}"] for name in METRIC_NAMES}
    result = {
        "model_dir": str(run.model_dir),
        "checkpoint": str(run.checkpoint),
        "checkpoint_epoch": metadata["epoch"],
        "checkpoint_global_step": metadata["global_step"],
        "experiment": str(run.experiment),
        "data_root": str(run.data_root),
        "info_path": str(run.info_path),
        "prediction_json": str(prediction_path.resolve()),
        "prediction_source": "provided_json" if provided_prediction is not None else "checkpoint_inference",
        "overall": overall,
        "metrics": metrics,
        "detailed_metrics": detailed,
        "counts": {"samples": len(sample_tokens), "gt": gt_counts,
                   "predictions": pred_counts, "ignored_gt": ignored_gt,
                   "ignored_predictions": ignored_pred, "missing_prediction_samples": 0},
        "protocol": {
            "classes": list(classes), "ap_center_distance_thresholds_m": run.distance_thresholds,
            "tp_center_distance_threshold_m": run.tp_distance_threshold,
            "min_recall": run.min_recall, "min_precision": run.min_precision,
            "score_threshold": run.score_thr, "max_boxes_per_sample": run.max_num,
            "nms_pre": run.nms_pre, "pcd_limit_range": run.pc_range,
            "matching_and_tp_metrics": "official nuscenes-devkit accumulate/calc_ap/calc_tp",
            "nds_formula": "(5*mAP + max(0,1-mATE) + max(0,1-mASE) + max(0,1-mAOE)) / 8",
            "note": "Rescaled nuScenes-style NDS; KITTI has no mAVE/mAAE annotations.",
        },
    }
    metric.write_json(run.output_dir / "metrics.json", result)
    with (run.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("class",) + METRIC_NAMES)
        writer.writeheader()
        for label, values in detailed.items():
            writer.writerow({"class": label, **{name: values[name] for name in METRIC_NAMES}})
    metric.print_metrics(run.model_dir.name, detailed, gt_counts, pred_counts)
    print(f"Metrics JSON: {run.output_dir / 'metrics.json'}")
    print(f"Metrics CSV:  {run.output_dir / 'metrics.csv'}")
    return overall


def main(argv=None):
    global metric
    args = parse_args(argv)
    metric = load_metric_module(args.metric_script)
    results = []
    for model_dir in args.model_dir:
        run, metadata = resolve_run(args, model_dir, batch=len(args.model_dir) > 1)
        print(f"\n[Selected] checkpoint: {run.checkpoint}", flush=True)
        print(f"[Selected] stored epoch={metadata['epoch']}, global_step={metadata['global_step']}")
        mode = metadata["hyper_parameters"].get("fusion_mode")
        model_name = f"HeightFusion {mode.upper()}" if mode else (
            "HeightFusion camera+LiDAR" if metadata["is_fusion"] else "BEVHeight++ camera")
        print(f"[Selected] model: {model_name}")
        print(f"[Selected] dataset: {run.data_root}\n[Selected] info: {run.info_path}")
        print(f"[Selected] experiment: {run.experiment}\n[Selected] output: {run.output_dir}", flush=True)
        if run.prediction_json is not None:
            print(f"[Selected] supplied predictions (no inference): {run.prediction_json}", flush=True)
        if not args.dry_run:
            results.append((run.model_dir, evaluate_run(run, metadata)))
    if results:
        print("\n| Model / dataset | mAP | NDS | mATE | mASE | mAOE |")
        print("| --- | --- | --- | --- | --- | --- |")
        for directory, overall in results:
            values = " | ".join(f"{overall[name]:.6f}" for name in METRIC_NAMES)
            print(f"| {directory.parent.name}/{directory.name} | {values} |")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
