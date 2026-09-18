"""Evaluate a trained checkpoint on the held-out test set.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt --model resnet18
"""

import argparse
import os

import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score
)

import config
from dataset import get_dataloaders
from model import build_model
from utils import get_device, load_checkpoint, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the tumor classifier.")
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
    parser.add_argument("--model", type=str, default="resnet18",
                         choices=["resnet18", "simple"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader, class_names = get_dataloaders(batch_size=args.batch_size)

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", class_names)

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(config.ASSETS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
