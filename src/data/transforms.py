"""
Model-agnostic image preprocessing and transform functions for IU X-Ray radiographs.
Supports aspect-ratio preserving letterboxing, direct bilinear resizing,
channel formatting, and PyTorch dataset integration.
"""

from typing import Tuple, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Loads an image from disk and verifies it."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {path}")
    with Image.open(path) as img:
        return img.convert("RGB")


def resize_with_aspect_ratio(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    fill_color: int = 0
) -> Image.Image:
    """
    Resizes image to fit within target_size while preserving aspect ratio,
    padding the remaining space with fill_color (letterboxing).
    """
    target_w, target_h = target_size
    orig_w, orig_h = image.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Determine mode for new canvas
    canvas = Image.new(image.mode, (target_w, target_h), (fill_color, fill_color, fill_color) if image.mode == "RGB" else fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))
    return canvas


def standard_resize(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """Standard direct bilinear resize without letterboxing."""
    return image.resize(target_size, Image.Resampling.BILINEAR)


class MedicalImageTransform:
    """
    Reusable, model-agnostic image transform pipeline.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        preserve_aspect_ratio: bool = True,
        to_rgb: bool = True,
        normalize_imagenet: bool = True
    ):
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.to_rgb = to_rgb
        self.normalize_imagenet = normalize_imagenet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.to_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        if self.preserve_aspect_ratio:
            image = resize_with_aspect_ratio(image, self.target_size)
        else:
            image = standard_resize(image, self.target_size)

        arr = np.array(image, dtype=np.float32) / 255.0

        if self.normalize_imagenet and self.to_rgb:
            arr = (arr - self.mean) / self.std

        # Convert [H, W, C] to [C, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor


class IUXRayDataset(Dataset):
    """
    PyTorch Dataset for IU X-Ray studies linking images, clinical queries, target reports, and labels.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Union[str, Path],
        transform: Optional[MedicalImageTransform] = None,
        image_col: str = "primary_image_filename",
        query_col: str = "clinical_query",
        target_col: str = "target_report"
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform or MedicalImageTransform()
        self.image_col = image_col
        self.query_col = query_col
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_name = str(row[self.image_col])
        image_path = self.images_dir / image_name

        image = load_image(image_path)
        tensor = self.transform(image)

        item = {
            "uid": int(row["uid"]),
            "image": tensor,
            "filename": image_name,
            "query": str(row.get(self.query_col, "")),
            "target_report": str(row.get(self.target_col, "")),
            "is_abnormal": int(row.get("is_abnormal", 0))
        }
        return item
