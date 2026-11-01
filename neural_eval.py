"""Held-out novel-view rendering and PSNR/SSIM/LPIPS evaluation."""

import argparse
import csv
from importlib.metadata import version
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .gaussian_scene import load_checkpoint, load_dataset, render


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("PSNR requires equal HxWx3 RGB images")
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def ssim(prediction: np.ndarray, target: np.ndarray, mask=None) -> float:
    """RGB channel-averaged, Gaussian-window SSIM on the full equal-resolution image."""
    from scipy.ndimage import gaussian_filter
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("SSIM requires equal HxWx3 RGB images")
    x, y = prediction.astype(np.float64), target.astype(np.float64)
    smooth = lambda value: gaussian_filter(value, sigma=(1.5, 1.5, 0), mode="reflect", truncate=10 / 3)
    ux, uy = smooth(x), smooth(y)
    vx = smooth(x * x) - ux * ux
    vy = smooth(y * y) - uy * uy
    vxy = smooth(x * y) - ux * uy
    c1, c2 = .01 ** 2, .03 ** 2
    value = ((2 * ux * uy + c1) * (2 * vxy + c2)
             / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))
    return float(value[mask].mean() if mask is not None else value.mean())


def evaluate(dataset_path: Path, checkpoint_path: Path, output: Path) -> dict:
    import lpips
    import torch
    dataset = load_dataset(dataset_path)
    trained = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if trained["dataset"] != str(dataset_path.resolve()):
        raise ValueError("Checkpoint and dataset do not match")
    params = load_checkpoint(checkpoint_path)
    metric = lpips.LPIPS(net="alex", version="0.1", verbose=False).cuda().eval()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    train = set(dataset["splits"]["train"])
    torch.cuda.synchronize()
    warmup_start = time.perf_counter()
    with torch.no_grad():
        render(params, dataset, dataset["frames"][dataset["splits"]["train"][0]])
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - warmup_start
    for group in ("validation", "test"):
        for index in dataset["splits"][group]:
            if index in train:
                raise ValueError("Held-out camera was in training data")
            frame = dataset["frames"][index]
            image = np.asarray(Image.open(dataset_path.parent / frame["image"]).convert("RGB"), dtype=np.float32) / 255.
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                predicted, alpha = render(params, dataset, frame)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            prediction = predicted.clamp(0, 1).cpu().numpy()
            target = image
            valid = np.load(dataset_path.parent / frame['mask']) if frame.get('mask') else np.ones(target.shape[:2], bool)
            # Exclude unobservable rectification borders; LPIPS sees identical neutral borders.
            prediction[~valid] = target[~valid]
            masked_mse = float(np.mean((prediction[valid].astype(np.float64)-target[valid])**2))
            pred_nchw = torch.tensor(prediction.transpose(2, 0, 1)[None], device="cuda") * 2 - 1
            true_nchw = torch.tensor(target.transpose(2, 0, 1)[None], device="cuda") * 2 - 1
            with torch.no_grad():
                lpips_value = float(metric(pred_nchw, true_nchw).item())
            stem = f"{group}_{index:04d}"
            Image.fromarray(np.rint(target * 255).astype(np.uint8)).save(output / f"{stem}_gt.png")
            Image.fromarray(np.rint(prediction * 255).astype(np.uint8)).save(output / f"{stem}_render.png")
            difference = np.abs(prediction - target)
            Image.fromarray(np.rint(difference * 255).astype(np.uint8)).save(output / f"{stem}_difference.png")
            (output / f"{stem}_camera.json").write_text(json.dumps({
                "index": index, "split": group, "timestamp_s": frame["timestamp_s"],
                "T_world_camera": frame["T_world_camera"], "T_camera_world": frame["T_camera_world"],
                "pose_source": dataset["pose_source"]}, indent=2))
            row = {"index": index, "split": group, "timestamp_s": frame["timestamp_s"],
                   "width": dataset["width"], "height": dataset["height"],
                   "valid_pixel_fraction": float(valid.mean()),
                   "psnr_db": float(-10*np.log10(max(masked_mse,1e-30))), "ssim": ssim(prediction, target, valid),
                   "lpips_alex_v0_1": lpips_value, "render_s": elapsed,
                   "render_fps": 1 / elapsed, "mean_alpha": float(alpha.mean().item())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (output / "per_view.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for group in ("validation", "test"):
        selected = [row for row in rows if row["split"] == group]
        aggregate[group] = {"views": len(selected),
                            "psnr_db": float(np.mean([r["psnr_db"] for r in selected])),
                            "ssim": float(np.mean([r["ssim"] for r in selected])),
                            "lpips_alex_v0_1": float(np.mean([r["lpips_alex_v0_1"] for r in selected])),
                            "mean_render_s": float(np.mean([r["render_s"] for r in selected])),
                            "fps_from_total_time": len(selected) / sum(r["render_s"] for r in selected)}
    report = {"checkpoint": str(checkpoint_path.resolve()), "checkpoint_reloaded": True,
              "dataset": str(dataset_path.resolve()), "pose_source": dataset["pose_source"],
              "depth_initialization": trained["depth_initialization"],
              "training_indices": dataset["splits"]["train"],
              "validation_indices": dataset["splits"]["validation"],
              "test_indices": dataset["splits"]["test"],
              "pixel_mask": "valid rectified RGB pixels for PSNR/SSIM; LPIPS full image with identical neutral invalid borders", "lpips_model": "AlexNet learned v0.1",
              "lpips_package_version": version("lpips"),
              "render_warmup_s_excluded": warmup_s,
              "aggregate": aggregate, "views": rows}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.checkpoint, args.output)["aggregate"], indent=2))


if __name__ == "__main__":
    main()
