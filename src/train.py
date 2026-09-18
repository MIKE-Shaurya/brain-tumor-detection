"""Train the brain tumor MRI classifier.

Usage:
    python src/train.py --epochs 15 --batch-size 32 --model resnet18
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_dataloaders, compute_class_weights
from model import build_model
from utils import set_seed, get_device, save_checkpoint, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train the tumor classifier.")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--model", type=str, default="resnet18",
                         choices=["resnet18", "simple"])
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="Freeze ResNet conv layers (resnet18 only).")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=args.batch_size
    )
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    class_weights = compute_class_weights(class_names).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_acc, class_names,
                             checkpoint_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    plot_training_curves(
        history, os.path.join(config.ASSETS_DIR, "training_curves.png")
    )
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training curves saved to: {config.ASSETS_DIR}/training_curves.png")


if __name__ == "__main__":
    main()
