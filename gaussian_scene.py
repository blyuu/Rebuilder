"""Small fixed-pose RGB-D initialized 3D Gaussian scene using gsplat."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .io import write_ply


def load_dataset(path: Path) -> dict:
    data = json.loads(path.read_text())
    frames = data["frames"]
    groups = data["splits"]
    all_indices = sum((groups[name] for name in ("train", "validation", "test")), [])
    if (sorted(all_indices) != list(range(len(frames))) or len(set(all_indices)) != len(frames)
            or any(frames[i]["split"] != group for group in groups for i in groups[group])):
        raise ValueError("Train/validation/test overlap or incomplete split")
    for frame in frames:
        pose = np.asarray(frame["T_world_camera"])
        inverse = np.asarray(frame["T_camera_world"])
        if not np.allclose(pose @ inverse, np.eye(4), atol=1e-5):
            raise ValueError("Camera pose conversion is inconsistent")
        if not (path.parent / frame["image"]).is_file():
            raise FileNotFoundError(frame["image"])
    return data


def initialize_rgbd(dataset: dict, root: Path, step: int = 5,
                    max_gaussians: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    """Backproject only training-frame TUM sensor depth into metric world points."""
    if step <= 0 or max_gaussians <= 0:
        raise ValueError("Positive sampling parameters required")
    k = dataset["intrinsics"]
    points, colors = [], []
    for index in dataset["splits"]["train"]:
        frame = dataset["frames"][index]
        depth = np.asarray(Image.open(root / frame["depth"]), dtype=np.float32) / dataset["depth_scale"]
        rgb = np.asarray(Image.open(root / frame["image"]).convert("RGB"), dtype=np.float32) / 255.
        yy, xx = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
        z = depth[::step, ::step]
        valid = np.isfinite(z) & (z >= .1) & (z <= 4.)
        xyz = np.stack(((xx - k["cx"]) * z / k["fx"],
                        (yy - k["cy"]) * z / k["fy"], z), axis=-1)[valid]
        pose = np.asarray(frame["T_world_camera"])
        points.append(xyz @ pose[:3, :3].T + pose[:3, 3])
        colors.append(rgb[::step, ::step][valid])
    xyz = np.concatenate(points).astype(np.float32)
    rgb = np.concatenate(colors).astype(np.float32)
    if len(xyz) > max_gaussians:
        indices = np.random.default_rng(2026).choice(len(xyz), max_gaussians, replace=False)
        xyz, rgb = xyz[indices], rgb[indices]
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("No valid RGB-D initialization points")
    return xyz, rgb


def camera_tensors(dataset: dict, frame: dict, device: str):
    import torch
    k = frame.get("intrinsics", dataset["intrinsics"])
    matrix = [[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]]
    return (torch.tensor(frame["T_camera_world"], dtype=torch.float32, device=device)[None],
            torch.tensor(matrix, dtype=torch.float32, device=device)[None])


def validate_render_output(image, width: int, height: int) -> None:
    if tuple(image.shape) != (1, height, width, 3):
        raise ValueError("Unexpected gsplat output resolution")
    if isinstance(image, np.ndarray) and not np.isfinite(image).all():
        raise ValueError("Nonfinite rendered RGB")


def render(parameters: dict, dataset: dict, frame: dict):
    import torch
    from gsplat import rasterization
    device = parameters["means"].device
    view, k = camera_tensors(dataset, frame, str(device))
    image, alpha, _ = rasterization(
        means=parameters["means"], quats=parameters["quats"],
        scales=torch.exp(parameters["log_scales"]).clamp(*dataset.get("scale_limits", [.002, .15])),
        opacities=torch.sigmoid(parameters["opacity_logits"]),
        colors=torch.sigmoid(parameters["color_logits"]),
        viewmats=view, Ks=k, width=dataset["width"], height=dataset["height"],
        packed=True, backgrounds=torch.tensor([.5, .5, .5], device=device))
    validate_render_output(image, dataset["width"], dataset["height"])
    return image[0], alpha[0]


def train(dataset_path: Path, output: Path, iterations: int = 400,
          sample_step: int = 5, max_gaussians: int = 12000) -> dict:
    import gsplat
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("3DGS training requires a working CUDA GPU")
    data = load_dataset(dataset_path)
    root = dataset_path.parent
    xyz, rgb = initialize_rgbd(data, root, sample_step, max_gaussians)
    output.mkdir(parents=True, exist_ok=True)
    write_ply(output / "initial_points.ply", xyz.astype(np.float64),
              np.rint(rgb * 255).astype(np.uint8))
    device = "cuda"
    torch.manual_seed(2026)
    params = {
        "means": torch.nn.Parameter(torch.tensor(xyz, device=device)),
        "log_scales": torch.nn.Parameter(torch.full((len(xyz), 3), np.log(.025), device=device)),
        "opacity_logits": torch.nn.Parameter(torch.full((len(xyz),), 0., device=device)),
        "color_logits": torch.nn.Parameter(torch.logit(torch.tensor(rgb, device=device).clamp(.01, .99))),
        "quats": torch.zeros((len(xyz), 4), device=device),
    }
    params["quats"][:, 0] = 1.
    optimizer = torch.optim.Adam([
        {"params": [params["means"]], "lr": .001},
        {"params": [params["log_scales"]], "lr": .003},
        {"params": [params["opacity_logits"]], "lr": .006},
        {"params": [params["color_logits"]], "lr": .012},
    ])
    targets = {}
    for index in data["splits"]["train"]:
        rgb_image = np.asarray(Image.open(root / data["frames"][index]["image"]).convert("RGB"), dtype=np.float32) / 255.
        targets[index] = torch.tensor(rgb_image, device=device)
    rng = np.random.default_rng(2026)
    history = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for iteration in range(iterations):
        index = int(rng.choice(data["splits"]["train"]))
        optimizer.zero_grad(set_to_none=True)
        prediction, alpha = render(params, data, data["frames"][index])
        target = targets[index]
        l1 = torch.mean(torch.abs(prediction - target))
        # Small opacity regularizer limits transparent floating splats.
        loss = l1 + .001 * torch.mean(torch.sigmoid(params["opacity_logits"]))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at iteration {iteration}")
        loss.backward()
        optimizer.step()
        history.append({"iteration": iteration + 1, "train_frame_index": index,
                        "loss": float(loss.item()), "l1": float(l1.item()),
                        "mean_alpha": float(alpha.detach().mean().item())})
        if (iteration + 1) % 25 == 0 or iteration == 0:
            print(f"iteration={iteration+1} loss={history[-1]['loss']:.5f} "
                  f"peak_gpu_bytes={torch.cuda.max_memory_allocated()}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    checkpoint = {"parameters": {name: value.detach().cpu() for name, value in params.items()},
                  "dataset": str(dataset_path.resolve()), "gaussians": len(xyz),
                  "pose_source": data["pose_source"], "depth_initialization": "TUM GT depth, train frames only",
                  "iterations": iterations}
    torch.save(checkpoint, output / "scene.pt")
    with (output / "training_loss.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {"framework": "gsplat", "gsplat_version": gsplat.__version__,
              "dataset": str(dataset_path.resolve()),
              "pose_source": data["pose_source"], "depth_initialization": checkpoint["depth_initialization"],
              "frames_train": len(data["splits"]["train"]), "resolution": [data["width"], data["height"]],
              "iterations": iterations, "gaussians": len(xyz), "sample_step": sample_step,
              "max_gaussians": max_gaussians, "initial_loss": history[0]["loss"],
              "final_loss": history[-1]["loss"], "elapsed_s": elapsed,
              "optimizer": {"name": "Adam", "means_lr": .001, "log_scales_lr": .003,
                            "opacity_logits_lr": .006, "color_logits_lr": .012},
              "loss": "full RGB L1 + 0.001 mean opacity",
              "rasterizer": {"gsplat_packed": True, "background_rgb": [.5, .5, .5],
                             "quaternion": "fixed identity", "color": "direct RGB logits, no SH",
                             "densification": False},
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
              "torch": torch.__version__, "cuda": torch.version.cuda}
    (output / "training.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def load_checkpoint(path: Path, device: str = "cuda") -> dict:
    import torch
    record = torch.load(path, map_location="cpu", weights_only=False)
    values = {name: value.to(device) for name, value in record["parameters"].items()}
    if len(values["means"]) != record["gaussians"]:
        raise ValueError("Checkpoint Gaussian count mismatch")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--max-gaussians", type=int, default=12000)
    args = parser.parse_args()
    train(args.dataset, args.output, args.iterations, args.sample_step, args.max_gaussians)


if __name__ == "__main__":
    main()
