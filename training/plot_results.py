import json
from pathlib import Path

import matplotlib.pyplot as plt


def count_images_by_class(root_dir, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    root = Path(root_dir)
    counts = {}
    if not root.exists():
        return counts

    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        n = sum(1 for f in class_dir.rglob("*") if f.suffix.lower() in exts)
        counts[class_dir.name] = n
    return counts


def load_metrics(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_class_distribution(train_dir, val_dir, test_dir, output_dir="models"):
    train_counts = count_images_by_class(train_dir)
    val_counts = count_images_by_class(val_dir)
    test_counts = count_images_by_class(test_dir)

    classes = sorted(set(train_counts) | set(val_counts) | set(test_counts))
    train_vals = [train_counts.get(c, 0) for c in classes]
    val_vals = [val_counts.get(c, 0) for c in classes]
    test_vals = [test_counts.get(c, 0) for c in classes]

    x = list(range(len(classes)))
    width = 0.28

    plt.figure(figsize=(28, 9))
    plt.bar([i - width for i in x], train_vals, width=width, label="Train")
    plt.bar(x, val_vals, width=width, label="Validation")
    plt.bar([i + width for i in x], test_vals, width=width, label="Test")
    plt.xticks(x, classes, rotation=90)
    plt.ylabel("Number of Images")
    plt.title("Dataset Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "class_distribution.png", dpi=200)
    plt.close()


def plot_training_curves(metrics, output_dir="models"):
    history = metrics.get("history", {})
    if not history:
        print("No history found in metrics.json")
        return

    stage1 = history.get("stage1", {})
    stage2 = history.get("stage2", {})

    stage1_epochs = list(range(1, len(stage1.get("train_loss", [])) + 1))
    stage2_epochs = list(range(len(stage1_epochs) + 1, len(stage1_epochs) + len(stage2.get("train_loss", [])) + 1))

    all_epochs = stage1_epochs + stage2_epochs

    train_loss = stage1.get("train_loss", []) + stage2.get("train_loss", [])
    val_loss = stage1.get("val_loss", []) + stage2.get("val_loss", [])
    train_acc = stage1.get("train_accuracy", []) + stage2.get("train_accuracy", [])
    val_acc = stage1.get("val_accuracy", []) + stage2.get("val_accuracy", [])

    train_acc = [x * 100 for x in train_acc]
    val_acc = [x * 100 for x in val_acc]

    stage_boundary = len(stage1_epochs)

    plt.figure(figsize=(11, 6))
    plt.plot(all_epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(all_epochs, val_loss, marker="o", label="Validation Loss")
    if stage_boundary > 0:
        plt.axvline(stage_boundary + 0.5, linestyle="--", label="Stage 1 -> Stage 2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(all_epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(all_epochs, val_acc, marker="o", label="Validation Accuracy")
    if stage_boundary > 0:
        plt.axvline(stage_boundary + 0.5, linestyle="--", label="Stage 1 -> Stage 2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "accuracy_curve.png", dpi=200)
    plt.close()


def plot_model_performance(metrics, output_dir="models"):
    val_acc = metrics.get("val_accuracy", 0) * 100
    test_acc = metrics.get("test_accuracy", 0) * 100
    gen_test_acc = metrics.get("generated_test_accuracy", 0) * 100
    target = metrics.get("config", {}).get("training", {}).get("target_accuracy", 0.85) * 100

    labels = ["Validation", "Test", "Generated Test", "Target"]
    values = [val_acc, test_acc, gen_test_acc, target]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, values)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Model Performance Summary")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "model_performance.png", dpi=200)
    plt.close()


def main():
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    plot_class_distribution("data/train", "data/val", "data/test", output_dir=output_dir)

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_metrics(metrics_path)
        plot_training_curves(metrics, output_dir=output_dir)
        plot_model_performance(metrics, output_dir=output_dir)
        print("Saved plots:")
        print("- models/class_distribution.png")
        print("- models/loss_curve.png")
        print("- models/accuracy_curve.png")
        print("- models/model_performance.png")
    else:
        print("Saved plot:")
        print("- models/class_distribution.png")
        print("metrics.json not found, so training/performance plots were skipped.")


if __name__ == "__main__":
    main()
