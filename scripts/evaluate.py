import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from openemotion.config import CLASS_NAMES, IMG_SIZE
from openemotion.model import load_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenEmotion model performance")
    parser.add_argument("--data", required=True, help="Path to FER2013 dataset directory")
    parser.add_argument("--model", required=True, help="Path to trained model .pth file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="evaluation", help="Directory to save evaluation metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    test_dir = Path(args.data) / "test" if (Path(args.data) / "test").exists() else Path(args.data)
    dataset = datasets.ImageFolder(str(test_dir), transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
    )

    model = load_model(args.model, device, len(CLASS_NAMES))

    labels = []
    predictions = []

    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device))
            batch_predictions = outputs.argmax(1).cpu()

            labels.extend(batch_labels.tolist())
            predictions.extend(batch_predictions.tolist())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = float(accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, average="macro"))

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    report_str = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
    )
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix — OpenEmotion")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    print("\nEvaluation Summary:")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
