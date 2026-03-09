"""
Configuration settings for AMR BioLinkBERT system.
"""
import os
import torch

# Model configuration
HF_TOKEN = ""
MODEL_NAME = "michiyasunaga/BioLinkBERT-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()

# Set environment
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# Data paths
DATA_PATH = "/kaggle/input/datasets/vihaankulkarni/antimicrobial-resistance-dataset/Kaggle_AMR_Dataset_v1.0.csv"
OUTPUT_DIR = "/kaggle/working"

# Training hyperparameters
EPOCHS = 5
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
MAX_LENGTH = 256
FILL_MASK_MAX_LENGTH = 512

# Ensemble weights
ZERO_SHOT_WEIGHT = 0.35
FINE_TUNED_WEIGHT = 0.65

# Vocabulary for zero-shot scoring
POSITIVE_WORDS = [
    "present", "detected", "identified", "confirmed", "positive",
    "found", "observed", "exists", "carried", "harboured",
    "expressed", "active", "yes", "true", "encoded",
]

NEGATIVE_WORDS = [
    "absent", "undetected", "missing", "negative", "lacking",
    "not", "no", "none", "lost", "inactive",
    "false", "unobserved", "absent", "rare",
]

# Visualization colors
VIZ_COLORS = {
    "BG": "#050c18",
    "CARD": "#0c1828",
    "TEXT": "#e6f0ff",
    "C1": "#38bdf8",
    "C2": "#f43f5e",
    "C3": "#fbbf24",
    "C4": "#34d399",
    "GRID": "#162032",
}

print(f"Device : {DEVICE}  |  GPUs available : {N_GPU}")
if N_GPU > 1:
    print(f"  → Will use DataParallel across {N_GPU} GPUs for fine-tuning")
