"""Run inference on a single MRI image.

Usage:
    python src/predict.py --checkpoint checkpoints/best_model.pt --image scan.jpg
"""

import argparse
import os

import torch
from PIL import Image

import config
from dataset import get_transforms
from model import build_model
from utils import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict tumor class for one image.")
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
    parser.add_argument("--model", type=str, default="resnet18",
                         choices=["resnet18", "simple"])
    parser.add_argument("--image", type=str, required=True,
                         help="Path to a single MRI image (jpg/png).")
    return parser.parse_args()


def predict(image_path, checkpoint_path, model_name="resnet18"):
    device = get_device()

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", config.CLASS_NAMES)

    model = build_model(model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_idx = int(probs.argmax())
    result = {
        "predicted_class": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        },
    }
    return result


def main():
    args = parse_args()
    result = predict(args.image, args.checkpoint, args.model)

    print(f"\nImage: {args.image}")
    print(f"Predicted class: {result['predicted_class']} "
          f"({result['confidence'] * 100:.2f}% confidence)")
    print("\nAll class probabilities:")
    for cls, prob in sorted(result["all_probabilities"].items(),
                             key=lambda x: -x[1]):
        print(f"  {cls:>12s}: {prob * 100:5.2f}%")


if __name__ == "__main__":
    main()
