"""Run CUDA monocular-depth inference and retain per-frame raw predictions."""

import argparse
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .depth_models import DA_REPO, MARIGOLD_REPO, adapter
from .io import load_scene


def _visualize(raw: np.ndarray, path: Path) -> None:
    low, high = np.percentile(raw, [2, 98])
    normalized = np.clip((raw - low) / max(high - low, 1e-6), 0, 1)
    Image.fromarray(np.rint(normalized * 255).astype(np.uint8)).save(path)


def run(manifest: Path, name: str, output: Path, *, processing_resolution: int = 512,
        steps: int = 4, seed: int = 2026, max_frames: int | None = None,
        checkpoint_path: str | None = None) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tensor execution is required for this experiment")
    scene = load_scene(manifest)
    frames = scene.frames[:max_frames]
    output.mkdir(parents=True, exist_ok=True)
    config = {"model": name, "checkpoint": DA_REPO if name == "depth_anything_v2_small" else MARIGOLD_REPO,
              "input_manifest": str(manifest.resolve()), "frames_requested": len(frames),
              "local_checkpoint_path": checkpoint_path,
              "processing_resolution": processing_resolution if name == "marigold_v1_1" else None,
              "diffusion_steps": steps if name == "marigold_v1_1" else None,
              "ensemble_size": 1 if name == "marigold_v1_1" else None,
              "seed": seed if name == "marigold_v1_1" else None,
              "raw_unit": "relative, nonmetric" if name == "depth_anything_v2_small"
                          else "affine-invariant normalized [0,1], nonmetric",
              "license": "Apache-2.0" if name == "depth_anything_v2_small"
                         else "CreativeML Open RAIL++-M",
              "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
              "python": platform.python_version()}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_load = time.perf_counter()
        options = {"device": "cuda"}
        if name == "marigold_v1_1":
            options.update(processing_resolution=processing_resolution, steps=steps, seed=seed)
            if checkpoint_path:
                options["checkpoint_path"] = checkpoint_path
        model = adapter(name, **options)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - start_load
        load_peak = torch.cuda.max_memory_allocated()
        results = []
        for index, frame in enumerate(frames):
            with Image.open(frame.rgb_path) as source:
                image = source.convert("RGB")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model.predict(image, index)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stem = f"frame_{index:04d}"
            np.save(output / f"{stem}_raw.npy", prediction, allow_pickle=False)
            _visualize(prediction, output / f"{stem}_raw.png")
            row = {"frame_index": index, "timestamp_s": frame.timestamp_s,
                   "rgb": str(frame.rgb_path), "raw_npy": f"{stem}_raw.npy",
                   "visualization_png": f"{stem}_raw.png", "shape": list(prediction.shape),
                   "raw_min": float(prediction.min()), "raw_max": float(prediction.max()),
                   "raw_median": float(np.median(prediction)), "inference_s": elapsed,
                   "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "gpu_free_after_bytes": torch.cuda.mem_get_info()[0]}
            results.append(row)
            print(json.dumps(row), flush=True)
        report = {**config, "checkpoint_revision": model.checkpoint_revision,
                  "preprocessor_size": dict(getattr(getattr(model, "processor", None), "size", {}) or {}),
                  "model_load_s": load_s, "model_load_peak_allocated_bytes": load_peak,
                  "frames_completed": len(results), "mean_inference_s": float(np.mean([r["inference_s"] for r in results])),
                  "median_inference_s": float(np.median([r["inference_s"] for r in results])),
                  "total_inference_s": float(sum(r["inference_s"] for r in results)),
                  "max_inference_gpu_allocated_bytes": max(r["gpu_peak_allocated_bytes"] for r in results),
                  "frames": results}
        (output / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception as error:
        failure = {**config, "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()}
        (output / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("model", choices=["depth_anything_v2_small", "marigold_v1_1"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processing-resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--checkpoint-path", type=str)
    args = parser.parse_args()
    result = run(args.manifest, args.model, args.output,
                 processing_resolution=args.processing_resolution,
                 steps=args.steps, seed=args.seed, max_frames=args.max_frames,
                 checkpoint_path=args.checkpoint_path)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
