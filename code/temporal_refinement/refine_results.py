#!/usr/bin/env python3
"""Temporally refine 3D detections stored in a pickle file.

The tool has no dataset, model, point-cloud, or framework dependency.  It reads
one result pickle and writes one result pickle while preserving the caller's
outer result structure and all non-geometric fields.
"""

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import pickle
from typing import Any


LABEL_FIELDS = ("label", "detection_name", "name")
SCORE_FIELDS = ("score", "detection_score")
CENTER_FIELDS = ("translation", "center")


@dataclass
class Observation:
    frame_index: int
    detection: dict[str, Any]
    center_field: str
    label: str
    score: float
    track_index: int | None = None


@dataclass
class Track:
    label: str
    observations: list[Observation] = field(default_factory=list)

    @property
    def last(self) -> Observation:
        return self.observations[-1]


def _value(detection: dict[str, Any], fields: tuple[str, ...], description: str) -> Any:
    for field in fields:
        if field in detection:
            return detection[field]
    raise ValueError(f"Detection is missing {description}; expected one of {fields}")


def _as_vector(value: Any, size: int, description: str) -> list[float]:
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must contain {size} numeric values") from error
    if len(vector) != size or not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{description} must contain {size} finite values")
    return vector


def _yaw_from_rotation(rotation: Any) -> float:
    w, x, y, z = _as_vector(rotation, 4, "rotation")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _set_rotation(detection: dict[str, Any], yaw: float) -> None:
    detection["rotation"] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _yaw(detection: dict[str, Any]) -> float:
    if "yaw" in detection:
        value = float(detection["yaw"])
        if not math.isfinite(value):
            raise ValueError("yaw must be finite")
        return value
    if "rotation" in detection:
        return _yaw_from_rotation(detection["rotation"])
    raise ValueError("Detection needs a finite yaw or a quaternion rotation")


def _set_yaw(detection: dict[str, Any], yaw: float) -> None:
    if "yaw" in detection:
        detection["yaw"] = yaw
    else:
        _set_rotation(detection, yaw)


def _frame_detections(payload: Any) -> list[list[dict[str, Any]]]:
    """Return mutable detection lists in the pickle's existing chronological order."""
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        frames = payload["frames"]
        result = []
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict) or not isinstance(frame.get("detections"), list):
                raise ValueError(f"frames[{index}] must contain a detections list")
            result.append(frame["detections"])
        return result
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        values = payload["results"]
        if not all(isinstance(detections, list) for detections in values.values()):
            raise ValueError("results must map frame identifiers to detection lists")
        return list(values.values())
    if isinstance(payload, list):
        result = []
        for index, frame in enumerate(payload):
            if not isinstance(frame, dict) or not isinstance(frame.get("detections"), list):
                raise ValueError(f"Input list item {index} must contain a detections list")
            result.append(frame["detections"])
        return result
    raise ValueError(
        "Unsupported pickle format. Use {'frames': [{'detections': [...]}]}, "
        "{'results': {frame_id: [...]}} or a list of frame dictionaries."
    )


def _observation(frame_index: int, detection: dict[str, Any]) -> Observation:
    if not isinstance(detection, dict):
        raise ValueError("Each detection must be a dictionary")
    center_field = next((field for field in CENTER_FIELDS if field in detection), None)
    if center_field is None:
        raise ValueError(f"Detection is missing center; expected one of {CENTER_FIELDS}")
    _as_vector(detection[center_field], 3, center_field)
    size = _as_vector(_value(detection, ("size",), "size"), 3, "size")
    if any(value <= 0.0 for value in size):
        raise ValueError("size values must be positive")
    _yaw(detection)
    label = str(_value(detection, LABEL_FIELDS, "label"))
    score = float(_value(detection, SCORE_FIELDS, "score"))
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return Observation(frame_index, detection, center_field, label, score)


def _distance(first: Observation, second: Observation) -> float:
    a = _as_vector(first.detection[first.center_field], 3, first.center_field)
    b = _as_vector(second.detection[second.center_field], 3, second.center_field)
    return math.dist(a, b)


def build_tracks(frames: list[list[dict[str, Any]]], association_distance: float) -> list[Track]:
    """Associate same-class detections greedily between adjacent frames."""
    tracks: list[Track] = []
    active: list[Track] = []
    for frame_index, detections in enumerate(frames):
        observations = [_observation(frame_index, detection) for detection in detections]
        candidates = []
        for observation in observations:
            for track in active:
                if track.label == observation.label:
                    candidates.append((_distance(track.last, observation), track, observation))
        assigned_tracks: set[int] = set()
        assigned_observations: set[int] = set()
        next_active: list[Track] = []
        for distance, track, observation in sorted(candidates, key=lambda item: item[0]):
            if distance > association_distance:
                break
            track_id, observation_id = id(track), id(observation)
            if track_id in assigned_tracks or observation_id in assigned_observations:
                continue
            track.observations.append(observation)
            assigned_tracks.add(track_id)
            assigned_observations.add(observation_id)
            next_active.append(track)
        for observation in observations:
            if id(observation) not in assigned_observations:
                track = Track(observation.label, [observation])
                tracks.append(track)
                next_active.append(track)
        active = next_active
    return tracks


def _weighted_mean(vectors: list[list[float]], weights: list[float]) -> list[float]:
    total = sum(weights)
    return [sum(vector[index] * weight for vector, weight in zip(vectors, weights)) / total
            for index in range(len(vectors[0]))]


def smooth_tracks(tracks: list[Track], window: int, min_track_length: int) -> None:
    radius = window // 2
    for track in tracks:
        if len(track.observations) < min_track_length:
            continue
        replacements: list[tuple[Observation, list[float], list[float], float]] = []
        for index, observation in enumerate(track.observations):
            nearby = track.observations[max(0, index - radius):index + radius + 1]
            weights = [max(item.score, 1e-6) for item in nearby]
            centers = [_as_vector(item.detection[item.center_field], 3, item.center_field) for item in nearby]
            sizes = [_as_vector(item.detection["size"], 3, "size") for item in nearby]
            yaws = [_yaw(item.detection) for item in nearby]
            sin_yaw = sum(math.sin(yaw) * weight for yaw, weight in zip(yaws, weights))
            cos_yaw = sum(math.cos(yaw) * weight for yaw, weight in zip(yaws, weights))
            replacements.append((observation, _weighted_mean(centers, weights),
                                 _weighted_mean(sizes, weights), math.atan2(sin_yaw, cos_yaw)))
        for observation, center, size, yaw in replacements:
            observation.detection[observation.center_field] = center
            observation.detection["size"] = size
            _set_yaw(observation.detection, yaw)


def refine(payload: Any, association_distance: float, window: int, min_track_length: int) -> Any:
    if association_distance <= 0.0 or not math.isfinite(association_distance):
        raise ValueError("association_distance must be finite and positive")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if min_track_length < 2:
        raise ValueError("min_track_length must be at least 2")
    frames = _frame_detections(payload)
    tracks = build_tracks(frames, association_distance)
    smooth_tracks(tracks, window, min_track_length)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input result pickle")
    parser.add_argument("--output", required=True, type=Path, help="Output refined result pickle")
    parser.add_argument("--association-distance", type=float, default=3.0,
                        help="Maximum center distance for same-class adjacent-frame association")
    parser.add_argument("--window", type=int, default=3, help="Odd temporal smoothing window")
    parser.add_argument("--min-track-length", type=int, default=2,
                        help="Minimum associated detections required before smoothing")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input pickle does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")
    with args.input.open("rb") as stream:
        payload = pickle.load(stream)
    refined = refine(payload, args.association_distance, args.window, args.min_track_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(refined, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Refined results written to {args.output}")


if __name__ == "__main__":
    main()
