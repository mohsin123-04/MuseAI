"""
MuseAI Configuration
Central configuration for all training and inference settings.
"""
import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Base paths - adjust these based on your environment
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = BASE_DIR / "datasets"  # Where raw images are stored
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Style data paths
STYLE_RAW_DIR = DATASETS_DIR  # Your existing datasets/picasso and datasets/rembrandt
STYLE_PROCESSED_DIR = DATA_DIR / "style"

# Content data paths  
CONTENT_RAW_DIR = DATA_DIR / "content_raw"
CONTENT_PROCESSED_DIR = DATA_DIR / "content" / "faces"

# =============================================================================
# ARTIST CONFIGURATION
# =============================================================================
ARTISTS = ["picasso", "rembrandt"]
ARTIST_TO_IDX = {"picasso": 0, "rembrandt": 1}
IDX_TO_ARTIST = {0: "picasso", 1: "rembrandt"}
NUM_ARTISTS = len(ARTISTS)

# =============================================================================
# IMAGE CONFIGURATION
# =============================================================================
IMAGE_SIZE = 512  # Output resolution
STYLE_IMAGE_SIZE = 512
CONTENT_IMAGE_SIZE = 512

# ImageNet normalization (for VGG)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# VGG Encoder settings
VGG_LAYERS = {
    'relu1_1': 'features.1',
    'relu2_1': 'features.6', 
    'relu3_1': 'features.11',
    'relu4_1': 'features.20',  # Main layer for AdaIN
    'relu5_1': 'features.29',
}
ENCODER_OUTPUT_LAYER = 'relu4_1'  # Layer to use for AdaIN

# Style embedding dimension
STYLE_EMBEDDING_DIM = 256

# Decoder architecture
DECODER_CHANNELS = [512, 256, 128, 64, 3]

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Basic training params
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WARMUP_EPOCHS = 2

# Loss weights
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 10.0
IDENTITY_WEIGHT = 5.0  # FaceNet identity preservation
TV_WEIGHT = 1e-6  # Total variation for smoothness

# Style loss layers
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
CONTENT_LAYERS = ['relu4_1']

# Optimizer settings
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-5

# Learning rate schedule
LR_DECAY_FACTOR = 0.5
LR_DECAY_EPOCHS = [20, 35, 45]

# =============================================================================
# DATA LOADING
# =============================================================================
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# =============================================================================
# FACE DETECTION CONFIGURATION
# =============================================================================
# MTCNN settings for face detection
FACE_DETECTION_THRESHOLD = 0.9
FACE_BBOX_EXPAND_RATIO = 1.4  # Expand bounding box by 40%
MIN_FACE_SIZE = 50

# =============================================================================
# FACENET IDENTITY CONFIGURATION
# =============================================================================
FACENET_MODEL = 'vggface2'  # Pre-trained model: 'vggface2' or 'casia-webface'
FACENET_INPUT_SIZE = 160
IDENTITY_EMBEDDING_DIM = 512

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
DEFAULT_STYLE_STRENGTH = 1.0  # 0.0 = content only, 1.0 = full style
INFERENCE_BATCH_SIZE = 1

# =============================================================================
# CHECKPOINTING
# =============================================================================
SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
KEEP_LAST_N_CHECKPOINTS = 3

# =============================================================================
# LOGGING
# =============================================================================
LOG_FREQUENCY = 100  # Log every N batches
SAMPLE_FREQUENCY = 500  # Save sample outputs every N batches

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
USE_AMP = True  # Automatic Mixed Precision for faster training

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_directories():
    """Create all necessary directories."""
    dirs = [
        DATA_DIR,
        STYLE_PROCESSED_DIR / "picasso" / "train",
        STYLE_PROCESSED_DIR / "picasso" / "val", 
        STYLE_PROCESSED_DIR / "picasso" / "test",
        STYLE_PROCESSED_DIR / "rembrandt" / "train",
        STYLE_PROCESSED_DIR / "rembrandt" / "val",
        STYLE_PROCESSED_DIR / "rembrandt" / "test",
        CONTENT_PROCESSED_DIR / "train",
        CONTENT_PROCESSED_DIR / "val",
        CONTENT_PROCESSED_DIR / "test",
        CHECKPOINTS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        BASE_DIR / "metadata",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Created {len(dirs)} directories")

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        if NUM_GPUS > 1:
            print(f"Multiple GPUs available: {NUM_GPUS}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

if __name__ == "__main__":
    create_directories()
    print(f"\nConfiguration loaded:")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Device: {DEVICE}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Artists: {ARTISTS}")
