from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data_utils import build_dataloaders, load_config, save_labels, save_metrics, set_seed
from model import build_model, unfreeze_top_layers
from tuning import build_trial_configs


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def train_epochs(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience):
    best_state = None
    best_val_acc = -1.0
    patience_left = patience
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=float(loss.item()))

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_once(config: dict, trial_index: int, device: torch.device):
    set_seed(config["training"].get("seed", 42) + trial_index)
    train_loader, val_loader, test_loader, generated_test_loader, class_names, class_weights = build_dataloaders(config)

    model = build_model(config, num_classes=len(class_names), train_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=float(config["model"].get("label_smoothing", 0.0)))

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["training"]["learning_rate_stage1"]),
    )
    history_stage1 = train_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=int(config["training"].get("epochs_stage1", 6)),
        patience=int(config["training"].get("early_stopping_patience", 3)),
    )

    unfreeze_top_layers(model, config["model"].get("backbone", "efficientnet_b0"), int(config["model"].get("unfreeze_last_blocks", 2)))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["training"]["learning_rate_stage2"]),
    )
    history_stage2 = train_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=int(config["training"].get("epochs_stage2", 6)),
        patience=int(config["training"].get("early_stopping_patience", 3)),
    )

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    gen_loss, gen_acc = evaluate(model, generated_test_loader, criterion, device)

    metrics = {
        "trial": trial_index,
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "generated_test_loss": float(gen_loss),
        "generated_test_accuracy": float(gen_acc),
        "classes": class_names,
        "history": {"stage1": history_stage1, "stage2": history_stage2},
        "config": config,
    }
    return model, metrics, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 15-class animal image classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    base_config = load_config(args.config)
    trial_configs = build_trial_configs(base_config)
    target_accuracy = float(base_config["training"].get("target_accuracy", 0.85))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_metrics = None
    best_acc = -1.0

    for i, trial_config in enumerate(trial_configs, start=1):
        print(f"\n=== Trial {i}/{len(trial_configs)} ===")
        print(json.dumps({
            "learning_rate_stage1": trial_config["training"]["learning_rate_stage1"],
            "dropout_rate": trial_config["model"]["dropout_rate"],
            "dense_units": trial_config["model"]["dense_units"],
        }, indent=2))

        model, metrics, class_names = train_once(trial_config, i, device)
        current_acc = metrics["val_accuracy"]

        if current_acc > best_acc:
            best_acc = current_acc
            best_metrics = metrics
            out_path = Path(base_config["output"]["model_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "config": base_config,
            }, out_path)
            save_labels(class_names, base_config["output"]["labels_path"])
            save_metrics(best_metrics, base_config["output"]["metrics_path"])
            print(f"Saved new best model with val_accuracy={best_acc:.4f}")

        if current_acc >= target_accuracy:
            print(f"Target accuracy reached: {current_acc:.4f} >= {target_accuracy:.2f}")
            break

    if best_metrics is None:
        raise RuntimeError("Training failed: no successful trial completed.")

    print("\nTraining complete.")
    print(json.dumps({
        "best_val_accuracy": best_metrics["val_accuracy"],
        "best_test_accuracy": best_metrics["test_accuracy"],
        "best_generated_test_accuracy": best_metrics["generated_test_accuracy"],
        "model_path": base_config["output"]["model_path"],
        "labels_path": base_config["output"]["labels_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
