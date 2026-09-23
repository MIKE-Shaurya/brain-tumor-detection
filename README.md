# 🧠 BrainSight — MRI Tumor Classifier 

A PyTorch deep learning project that classifies brain MRI scans into four
categories — **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**
— using a convolutional neural network. Built as a portfolio project to
demonstrate an end-to-end image classification pipeline: data loading,
model training, evaluation, and single-image inference.

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is
> **not** a medical device and must never be used for real clinical
> diagnosis or patient care decisions.

---

## 📂 Project Structure

```
brain-tumor-detection/
├── data/                   # (not committed) place dataset here
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
├── src/
│   ├── config.py            # Central config (paths, hyperparameters)
│   ├── dataset.py           # Dataset loading & transforms
│   ├── model.py             # CNN architectures (custom + transfer learning)
│   ├── train.py              # Training loop with checkpointing
│   ├── evaluate.py          # Test-set evaluation, confusion matrix, metrics
│   ├── predict.py           # Run inference on a single image
│   └── utils.py             # Seeding, checkpoint I/O, plotting helpers
├── notebooks/
│   └── exploration.ipynb    # (optional) EDA / experiments scratchpad
├── assets/                  # Saved plots (confusion matrix, curves) go here
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

This project expects the **Brain Tumor MRI Dataset** (Kaggle), which has
~7,000 labeled MRI images split across 4 classes, already organized in
`train/`  and `test/` folders per class — a standard `ImageFolder` layout.

- Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Download it, unzip, and place it so the paths match the structure above
(or edit `DATA_DIR` in `src/config.py`).

## 🚀 Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset under data/train and data/test (see above)

# 5. Train
python src/train.py --epochs 15 --batch-size 32 --model resnet18

# 6. Evaluate on the test set
python src/evaluate.py --checkpoint checkpoints/best_model.pt

# 7. Predict on a single image
python src/predict.py --checkpoint checkpoints/best_model.pt --image path/to/scan.jpg
```

## 🏗️ Model

Two architecture options are included in `src/model.py`:

1. **`SimpleCNN`** — a small custom CNN (4 conv blocks) built from scratch,
   useful for understanding the fundamentals and fast experimentation on a CPU.
2. **`ResNetTransfer`** — a `torchvision` ResNet18 backbone pretrained on
   ImageNet with a replaced classification head, fine-tuned on the MRI data.
   This is the recommended option for best accuracy.

Select which one to train with `--model simple` or `--model resnet18`.

## 📈 Results

After training, `evaluate.py` writes:
- Overall accuracy, precision, recall, F1 (per class + macro average)
- A confusion matrix image to `assets/confusion_matrix.png`
- Training/validation loss & accuracy curves to `assets/training_curves.png`

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Glioma | 0.98 | 0.84 | 0.90 |
| Meningioma | 0.90 | 0.97 | 0.94 |
| No Tumor | 0.95 | 1.00 | 0.97 |
| Pituitary | 0.99 | 1.00 | 0.99 ||

| Metric | Score | 
| ---|---|
| Test Accuracy |  96.2% |
| Macro F1 |  0.95 |

## 🧪 Key Techniques Used

- Transfer learning (ResNet18 fine-tuning) vs. training from scratch
- Data augmentation (random flips/rotations) to reduce overfitting
- Stratified train/val split with early stopping on validation loss
- Class-weighted loss to handle mild class imbalance
- Reproducible runs via fixed random seeds
- Checkpointing of the best model by validation accuracy

## 🔧 Possible Extensions

- Add Grad-CAM visualizations to highlight what the model focuses on
- Swap in EfficientNet / DenseNet backbones and compare
- Wrap `predict.py` in a small Flask/FastAPI service or Streamlit demo
- Export to ONNX / TorchScript for deployment

## 📄 License

MIT — see `LICENSE`.
