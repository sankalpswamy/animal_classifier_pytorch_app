from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "training"))
from model import build_model  # noqa: E402


class AnimalPredictor:
    def __init__(self, model_path: str = "models/best_model.pt", labels_path: str = "models/labels.txt"):
        model_file = Path(model_path)
        labels_file = Path(labels_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found at {labels_path}. Train the model first.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = [line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        checkpoint = torch.load(model_file, map_location=self.device)
        self.config = checkpoint.get("config", {"data": {"image_size": 224}, "model": {"backbone": "efficientnet_b0", "dropout_rate": 0.3, "dense_units": 256}})
        self.model = build_model(self.config, num_classes=len(self.labels), train_backbone=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        image_size = int(self.config.get("data", {}).get("image_size", 224))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        with torch.no_grad():
            logits = self.model(self.preprocess(image))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(self.labels[i], float(probs[i])) for i in top_indices]

    def load_metrics(self, metrics_path: str = "models/metrics.json") -> dict:
        metrics_file = Path(metrics_path)
        if not metrics_file.exists():
            return {}
        return json.loads(metrics_file.read_text(encoding="utf-8"))
