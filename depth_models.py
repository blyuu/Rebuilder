"""CUDA adapters for relative monocular depth checkpoints."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


DA_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
MARIGOLD_REPO = "prs-eth/marigold-depth-v1-1"
MARIGOLD_REVISION = "9571e7123e258cf052b4e54241f17971c290e9a8"


def validate_prediction(prediction: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    value = np.asarray(prediction, dtype=np.float32)
    if value.shape != (size[1], size[0]) or not np.isfinite(value).all():
        raise ValueError(f"Expected finite depth shape {(size[1], size[0])}, got {value.shape}")
    if value.max() - value.min() < 1e-6:
        raise ValueError("Degenerate constant depth prediction")
    return value


@dataclass
class DepthAnythingV2Small:
    device: str = "cuda"

    def __post_init__(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from .app_runtime import model_path
        checkpoint = model_path('depth_anything_v2_small', DA_REPO)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device).eval()
        self.checkpoint_revision = getattr(self.model.config, "_commit_hash", None)

    def predict(self, image: Image.Image, frame_index: int = 0) -> np.ndarray:
        import torch
        values = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            prediction = self.model(**values).predicted_depth
            resized = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(image.height, image.width),
                mode="bicubic", align_corners=False)[0, 0]
        return validate_prediction(resized.float().cpu().numpy(), image.size)


@dataclass
class MarigoldV11:
    device: str = "cuda"
    checkpoint_path: str = MARIGOLD_REPO
    processing_resolution: int = 512
    steps: int = 4
    seed: int = 2026

    def __post_init__(self):
        import torch
        from diffusers import MarigoldDepthPipeline
        from .app_runtime import model_path
        self.checkpoint_path = model_path('marigold_v1_1', self.checkpoint_path)
        self.pipe = MarigoldDepthPipeline.from_pretrained(
            self.checkpoint_path, variant="fp16", dtype=torch.float16).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.checkpoint_revision = (MARIGOLD_REVISION if self.checkpoint_path != MARIGOLD_REPO
                                    else getattr(self.pipe.config, "_commit_hash", None))

    def predict(self, image: Image.Image, frame_index: int = 0) -> np.ndarray:
        import torch
        generator = torch.Generator(device=self.device).manual_seed(self.seed + frame_index)
        with torch.inference_mode():
            output = self.pipe(image.convert("RGB"), num_inference_steps=self.steps,
                               ensemble_size=1, processing_resolution=self.processing_resolution,
                               match_input_resolution=True, generator=generator,
                               output_type="np")
        return validate_prediction(np.asarray(output.prediction[0, :, :, 0]), image.size)


def adapter(name: str, **kwargs):
    if name == "depth_anything_v2_small":
        return DepthAnythingV2Small(**kwargs)
    if name == "marigold_v1_1":
        return MarigoldV11(**kwargs)
    raise ValueError(f"Unknown model: {name}")
