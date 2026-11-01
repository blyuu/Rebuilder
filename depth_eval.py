"""Oracle metric alignment and held-out depth evaluation for nonmetric models."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .io import load_rgbd, load_scene


def valid_depth_mask(depth_m: np.ndarray, min_m: float = .1, max_m: float = 4.) -> np.ndarray:
    return np.isfinite(depth_m) & (depth_m >= min_m) & (depth_m <= max_m)


def encode_metric_depth(depth: np.ndarray, *, unit: str, scale: float = 5000.,
                        min_m: float = .1, max_m: float = 4.) -> np.ndarray:
    if unit != "meters":
        raise ValueError("TSDF depth export requires meters, not a relative model output")
    if not 0 < min_m < max_m or scale <= 0 or max_m * scale > np.iinfo(np.uint16).max:
        raise ValueError("Invalid metric depth encoding range")
    valid = valid_depth_mask(depth, min_m, max_m)
    return np.rint(np.where(valid, depth, 0) * scale).astype(np.uint16)


def depth_signal(name: str, raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if name == "depth_anything_v2_small":
        # Its relative activation is larger for nearer pixels. Reverse its
        # direction using only this prediction, retaining nonmetric units.
        return np.max(raw) - raw + 1e-3
    if name == "marigold_v1_1":
        # Marigold v1.1 predicts an affine-invariant near=0, far=1 map.
        return raw
    raise ValueError(name)


def fit_alignment(signal: np.ndarray, gt_m: np.ndarray, mask: np.ndarray,
                  method: str) -> tuple[float, float]:
    x = np.asarray(signal[mask], dtype=np.float64)
    y = np.asarray(gt_m[mask], dtype=np.float64)
    if len(x) < 20 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("At least 20 finite calibration points required")
    if method == "scale_only":
        a = float(np.dot(x, y) / np.dot(x, x))
        b = 0.0
    elif method in {"scale_shift", "robust"}:
        design = np.column_stack((x, np.ones_like(x)))
        a, b = np.linalg.lstsq(design, y, rcond=None)[0]
        if method == "robust":
            for _ in range(8):
                residual = y - (a*x + b)
                sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
                threshold = max(1e-6, 1.345 * sigma)
                weights = np.minimum(1.0, threshold / np.maximum(np.abs(residual), 1e-12))
                weighted = design * np.sqrt(weights)[:, None]
                a, b = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)[0]
        a, b = float(a), float(b)
    else:
        raise ValueError(method)
    if not np.isfinite([a, b]).all() or a <= 0:
        raise ValueError(f"Nonphysical alignment scale: {a}")
    return a, b


def depth_metrics(pred_m: np.ndarray, gt_m: np.ndarray, mask: np.ndarray) -> dict:
    pred = np.asarray(pred_m[mask], dtype=np.float64)
    gt = np.asarray(gt_m[mask], dtype=np.float64)
    if not len(pred) or not np.isfinite(pred).all() or np.any(gt <= 0):
        raise ValueError("Metrics require finite predictions and positive GT")
    ratio = np.full_like(pred, np.inf)
    positive = pred > 0
    ratio[positive] = np.maximum(pred[positive] / gt[positive], gt[positive] / pred[positive])
    return {"pixels": len(pred), "absrel": float(np.mean(np.abs(pred-gt)/gt)),
            "rmse_m": float(np.sqrt(np.mean((pred-gt)**2))),
            "delta1": float(np.mean(ratio < 1.25)),
            "positive_prediction_fraction": float(np.mean(positive)),
            "absolute_relative_error_sum": float(np.sum(np.abs(pred-gt)/gt)),
            "squared_error_sum_m2": float(np.sum((pred-gt)**2)),
            "delta1_correct_pixels": int(np.count_nonzero(ratio < 1.25))}


def run(manifest: Path, prediction_root: Path, output: Path,
        calibration_stride: int = 4, min_depth_m: float = .1,
        max_depth_m: float = 4.) -> dict:
    scene = load_scene(manifest)
    names = ("depth_anything_v2_small", "marigold_v1_1")
    methods = ("scale_only", "scale_shift", "robust")
    reports = {name: json.loads((prediction_root / name / "inference.json").read_text()) for name in names}
    if any(report["frames_completed"] != len(scene.frames) for report in reports.values()):
        raise ValueError("Models must predict every identical manifest frame")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest_frames = {name: [] for name in names}
    for index, frame in enumerate(scene.frames):
        _, gt = load_rgbd(frame, scene)
        raw = {name: np.load(prediction_root / name / f"frame_{index:04d}_raw.npy") for name in names}
        if any(p.shape != gt.shape for p in raw.values()):
            raise ValueError("Prediction and reference dimensions differ")
        signal = {name: depth_signal(name, raw[name]) for name in names}
        common = valid_depth_mask(gt, min_depth_m, max_depth_m)
        for name in names:
            common &= np.isfinite(signal[name]) & (signal[name] > 0)
        yy, xx = np.indices(gt.shape)
        calibration = common & ((xx % calibration_stride == 0) & (yy % calibration_stride == 0))
        evaluation = common & ~calibration
        if not calibration.any() or not evaluation.any():
            raise ValueError("No calibration or held-out pixels")
        row = {"frame_index": index, "timestamp_s": frame.timestamp_s,
               "common_valid_pixels": int(common.sum()),
               "oracle_calibration_pixels": int(calibration.sum()),
               "heldout_evaluation_pixels": int(evaluation.sum()), "models": {}}
        for name in names:
            model_dir = output / name
            model_dir.mkdir(exist_ok=True)
            model_row = {"raw": {"unit": reports[name]["raw_unit"], "absrel": None,
                                 "rmse_m": None, "delta1": None,
                                 "min": float(raw[name].min()), "max": float(raw[name].max())}}
            for method in methods:
                a, b = fit_alignment(signal[name], gt, calibration, method)
                aligned = (a * signal[name] + b).astype(np.float32)
                metrics = depth_metrics(aligned, gt, evaluation)
                model_row[method] = {"alignment": "oracle_gt_calibration_pixels",
                                     "a": a, "b_m": b, "unit": "meters after oracle alignment",
                                     **metrics}
                stem = f"frame_{index:04d}_{method}"
                np.save(model_dir / f"{stem}_meters.npy", aligned, allow_pickle=False)
                error = np.full(gt.shape, np.nan, dtype=np.float32)
                error[evaluation] = np.abs(aligned[evaluation] - gt[evaluation])
                np.save(model_dir / f"{stem}_heldout_abs_error_m.npy", error, allow_pickle=False)
                preview = np.clip((aligned - min_depth_m) / (max_depth_m - min_depth_m), 0, 1)
                Image.fromarray(np.rint(preview * 255).astype(np.uint8)).save(model_dir / f"{stem}_meters.png")
                err_preview = np.clip(np.nan_to_num(error, nan=0.0) / .5, 0, 1)
                Image.fromarray(np.rint(err_preview * 255).astype(np.uint8)).save(model_dir / f"{stem}_heldout_error.png")
                if method == "robust":
                    # Keep the sensor's valid support fixed across TSDF conditions.
                    # Out-of-range model pixels remain invalid rather than clipped.
                    encoded = encode_metric_depth(aligned, unit="meters", scale=5000,
                                                  min_m=min_depth_m, max_m=max_depth_m)
                    encoded[~valid_depth_mask(gt, min_depth_m, max_depth_m)] = 0
                    depth_path = model_dir / f"frame_{index:04d}_tsdf_depth.png"
                    Image.fromarray(encoded).save(depth_path)
                    manifest_frames[name].append({**json.loads(manifest.read_text())["frames"][index],
                                                   "rgb": str(frame.rgb_path.resolve()),
                                                   "depth": str(depth_path.resolve()),
                                                   "depth_timestamp_s": frame.timestamp_s,
                                                   "depth_source": f"{name}_oracle_robust"})
            row["models"][name] = model_row
        rows.append(row)
    for name in names:
        original = json.loads(manifest.read_text())
        aligned_manifest = {**original, "depth_scale": 5000.0,
                            "source": {**original.get("source", {}),
                                       "depth_source": name,
                                       "depth_alignment": "per-frame oracle robust GT sample 1/16 pixels"},
                            "frames": manifest_frames[name]}
        (output / name / "tsdf_manifest.json").write_text(json.dumps(aligned_manifest, indent=2))
    aggregate = {}
    for name in names:
        aggregate[name] = {"raw": {"unit": reports[name]["raw_unit"],
                                   "absrel": None, "rmse_m": None, "delta1": None}}
        for method in methods:
            values = [row["models"][name][method] for row in rows]
            pixels = sum(v["pixels"] for v in values)
            aggregate[name][method] = {"unit": "meters after oracle alignment",
                "pixels": pixels,
                "absrel": sum(v["absolute_relative_error_sum"] for v in values) / pixels,
                "rmse_m": float(np.sqrt(sum(v["squared_error_sum_m2"] for v in values) / pixels)),
                "delta1": sum(v["delta1_correct_pixels"] for v in values) / pixels}
    report = {"input_manifest": str(manifest.resolve()), "frames": len(rows),
              "calibration_stride": calibration_stride,
              "calibration_fraction_of_grid": 1/calibration_stride**2,
              "valid_depth_range_m": [min_depth_m, max_depth_m],
              "oracle_alignment": True,
              "evaluation_overlap_with_calibration_pixels": False,
              "same_frame_gt_used_for_calibration_and_evaluation": True,
              "common_heldout_evaluation_pixels": sum(r["heldout_evaluation_pixels"] for r in rows),
              "aggregate": aggregate, "frames_detail": rows}
    (output / "depth_metrics.json").write_text(json.dumps(report, indent=2))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "method", "unit", "heldout_pixels", "absrel", "rmse_m", "delta1"])
        for name in names:
            for method in ("raw", *methods):
                value = aggregate[name][method]
                writer.writerow([name, method, value["unit"], value.get("pixels", ""),
                                 value["absrel"], value["rmse_m"], value["delta1"]])
    with (output / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "timestamp_s", "model", "method", "heldout_pixels",
                         "oracle_scale", "oracle_shift_m", "absrel", "rmse_m", "delta1"])
        for row in rows:
            for name in names:
                for method in methods:
                    value = row["models"][name][method]
                    writer.writerow([row["frame_index"], row["timestamp_s"], name, method,
                                     value["pixels"], value["a"], value["b_m"],
                                     value["absrel"], value["rmse_m"], value["delta1"]])
    return {k: v for k, v in report.items() if k != "frames_detail"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("prediction_root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration-stride", type=int, default=4)
    args = p.parse_args()
    print(json.dumps(run(args.manifest, args.prediction_root, args.output,
                         args.calibration_stride), indent=2))
