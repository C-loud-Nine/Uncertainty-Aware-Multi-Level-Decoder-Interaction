"""Configuration file for multi-task breast ultrasound model."""

import os

# ============================================================================
# PATHS
# ============================================================================
BUSI_PATH = './data/BUSI'
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
INPUT_SIZE = (224, 224, 3)
DECODER_CHANNELS = [384, 192, 96, 48]
NUM_SEG_CLASSES = 1
NUM_CLF_CLASSES = 3
DROPOUT_RATE = 0.3

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 24
VAL_BATCH_SIZE = 24
EPOCHS = 100
WARMUP_EPOCHS = 3

# Learning rate
INITIAL_LR = 3.0e-4
MIN_LR = 1.5e-6
COSINE_ALPHA = 0.03

# Optimizer
ADAM_BETA_1 = 0.91
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-07
GLOBAL_CLIPNORM = 1.0

# ============================================================================
# DATA SPLIT
# ============================================================================
TRAIN_SPLIT = 0.70  # 70%
VAL_SPLIT = 0.15    # 15%
TEST_SPLIT = 0.15   # 15%
RANDOM_STATE = 42

# ============================================================================
# LOSS WEIGHTS
# ============================================================================
SEG_WEIGHT_START = 0.80
CLF_WEIGHT_START = 0.20
SEG_WEIGHT_FINAL = 0.78
CLF_WEIGHT_FINAL = 0.22

# ============================================================================
# CALLBACKS
# ============================================================================
EARLY_STOPPING_PATIENCE = 22
EARLY_STOPPING_MIN_DELTA = 0.004
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.55
REDUCE_LR_COOLDOWN = 2
MONITOR = "val_combined"
MODE = "max"

# ============================================================================
# AUGMENTATION
# ============================================================================
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ROTATION_LIMIT = 15
ROTATION_PROB = 0.7
NUM_AUGMENTATIONS = 3  # Generate 3 augmented versions per sample

# ============================================================================
# CATEGORIES
# ============================================================================
CATEGORIES = ['normal', 'benign', 'malignant']
