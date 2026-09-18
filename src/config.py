"""Central configuration for paths, classes, and default hyperparameters."""

import os

# --- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Classes -------------------------------------------------------------
# Must match the subfolder names under data/train and data/test
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Image / training defaults -----------------------------------------
IMAGE_SIZE = 224          # ResNet expects 224x224; SimpleCNN also uses this
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15          # fraction of train/ held out for validation
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5

# ImageNet normalization stats (used for both models so pretrained
# ResNet weights stay well-calibrated; SimpleCNN benefits too)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
