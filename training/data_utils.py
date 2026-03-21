from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(folder: Path, supported_extensions: list[str]) -> list[Path]:
    exts = {ext.lower() for ext in supported_extensions}
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def prepare_selected_dataset(config: dict) -> None:
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    selected_classes = config["project"]["selected_classes"]
    supported_extensions = data_cfg["supported_extensions"]

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset folder not found at {raw_dir}. Put the 15 class folders there before splitting.")

    summary = {}
    for class_name in selected_classes:
        source = raw_dir / class_name
        if not source.exists():
            raise FileNotFoundError(f"Missing selected class folder: {source}")
        images = list_images(source, supported_extensions)
        if len(images) < 10:
            raise ValueError(f"Class '{class_name}' has too few images ({len(images)}).")

        train_paths, temp_paths = train_test_split(
            images,
            test_size=(1.0 - data_cfg["train_split"]),
            random_state=config["training"]["seed"],
            shuffle=True,
        )
        relative_test = data_cfg["test_split"] / (data_cfg["val_split"] + data_cfg["test_split"])
        val_paths, test_paths = train_test_split(
            temp_paths,
            test_size=relative_test,
            random_state=config["training"]["seed"],
            shuffle=True,
        )

        for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            target_dir = Path(data_cfg[f"{split_name}_dir"]) / class_name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                shutil.copy2(src, target_dir / src.name)

        summary[class_name] = {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
            "total": len(images),
        }

    print(json.dumps(summary, indent=2))


def get_transforms(config: dict):
    image_size = int(config["data"]["image_size"])
    aug = config["augmentation"]
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(aug.get("random_resized_crop_scale_min", 0.85), 1.0)),
        transforms.RandomHorizontalFlip() if aug.get("horizontal_flip", True) else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(aug.get("rotation_degrees", 15)),
        transforms.ColorJitter(
            brightness=aug.get("color_jitter_brightness", 0.2),
            contrast=aug.get("color_jitter_contrast", 0.2),
            saturation=aug.get("color_jitter_saturation", 0.2),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    generated_eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return train_transform, eval_transform, generated_eval_transform


def build_dataloaders(config: dict):
    data_cfg = config["data"]
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    train_tf, eval_tf, gen_tf = get_transforms(config)

    train_ds = datasets.ImageFolder(data_cfg["train_dir"], transform=train_tf)
    val_ds = datasets.ImageFolder(data_cfg["val_dir"], transform=eval_tf)
    test_ds = datasets.ImageFolder(data_cfg["test_dir"], transform=eval_tf)
    generated_test_ds = datasets.ImageFolder(data_cfg["test_dir"], transform=gen_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    generated_test_loader = DataLoader(generated_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_ds.classes
    class_weights = compute_class_weights(Path(data_cfg["train_dir"]), class_names)
    return train_loader, val_loader, test_loader, generated_test_loader, class_names, class_weights


def compute_class_weights(train_dir: Path, class_names: list[str]) -> torch.Tensor:
    counts = Counter()
    for idx, class_name in enumerate(class_names):
        counts[idx] = len([p for p in (train_dir / class_name).iterdir() if p.is_file()])
    total = sum(counts.values())
    num_classes = len(class_names)
    weights = [total / (num_classes * counts[idx]) for idx in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def save_labels(labels: list[str], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text("\n".join(labels) + "\n", encoding="utf-8")


def save_metrics(metrics: dict, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
