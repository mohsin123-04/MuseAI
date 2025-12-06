# MuseAI - Identity-Preserving Neural Style Transfer

Transform portrait photos into artistic renditions in the styles of **Picasso** and **Rembrandt** while preserving facial identity.

## 🎯 Overview

MuseAI is an MVP neural style transfer system that:
- ✅ Accepts portrait uploads (selfies, photos)
- 🎨 Applies artistic styles (Picasso or Rembrandt)
- 👤 Preserves facial identity using FaceNet embeddings
- 🎨 Uses AdaIN (Adaptive Instance Normalization) for style mixing
- 🚀 Provides a web interface via Gradio
- 📊 Supports evaluation metrics (SSIM, LPIPS, Gram distance, identity similarity)

## 🏗️ Architecture

```
Input Portrait
     ↓
[VGG19 Encoder] → Extract content features (frozen)
     ↓
[AdaIN Layer] → Transfer style statistics to content
     ↓
[Decoder] → Reconstruct stylized portrait (trainable)
     ↓
Output Styled Portrait + FaceNet Identity Preservation
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **VGG19 Encoder** | Extract multi-level features (frozen, pretrained on ImageNet) |
| **AdaIN** | Transfer style statistics while preserving content structure |
| **Conditional AdaIN** | Artist-specific style modulation (Picasso vs Rembrandt) |
| **Decoder** | Reconstruct image from stylized features |
| **FaceNet** | Ensure identity preservation in styled portrait |
| **Loss Functions** | Content + Style + Identity + TV smoothness |

## 📁 Project Structure

```
MuseAI/
├── data/                           # Processed datasets
│   ├── style/                      # Processed paintings
│   │   ├── picasso/{train,val,test}/
│   │   └── rembrandt/{train,val,test}/
│   └── content/faces/{train,val,test}/
├── datasets/                       # Raw images (your provided data)
│   ├── picasso/                    # 170 Picasso paintings
│   └── rembrandt/                  # 200 Rembrandt paintings
├── checkpoints/                    # Saved model weights
├── logs/                           # Training logs & metrics
├── metadata/                       # Dataset catalogs (CSV)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration (paths, hyperparams)
│   ├── train.py                    # Training script
│   ├── inference.py                # Inference engine
│   ├── models/
│   │   ├── encoder.py              # VGG19 encoder
│   │   ├── adain.py                # AdaIN layers
│   │   ├── decoder.py              # Decoder network
│   │   ├── style_transfer.py       # Complete model + losses
│   │   ├── identity.py             # FaceNet identity module
│   │   └── losses.py               # Loss functions
│   ├── preprocess/
│   │   ├── style_preprocess.py     # Preprocess paintings
│   │   └── content_preprocess.py   # Preprocess faces (MTCNN)
│   └── utils/
│       ├── data_loader.py          # PyTorch datasets
│       └── metrics.py              # Evaluation metrics
├── app.py                          # Gradio web interface
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/mohsin123-04/MuseAI.git
cd MuseAI

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained models (automatic on first run)
# - VGG19 (ImageNet)
# - FaceNet (VGGFace2)
# - MTCNN (face detector)
```

### 2. Data Setup

Your Picasso and Rembrandt images are already in `datasets/`:
- `datasets/picasso/` (170 images)
- `datasets/rembrandt/` (200 images)

**Skip preprocessing for now** - the system will auto-load from `datasets/` folder.

### 3. Training (Optional)

```bash
# Preprocess style images (paintings)
python -m src.preprocess.style_preprocess

# Preprocess content images (faces) - requires face dataset
python -m src.preprocess.content_preprocess

# Start training
python src/train.py
```

**Training parameters** (edit in `src/config.py`):
- Batch size: 4
- Learning rate: 1e-4
- Epochs: 50
- Loss weights: Content=1.0, Style=10.0, Identity=5.0

### 4. Run the Web Interface

```bash
# Launch Gradio app
python app.py

# Open browser to: http://localhost:7860
```

## 🎮 Using the Interface

1. **Upload Portrait**: Click to upload a selfie or portrait photo
2. **Select Artist**: Choose Picasso or Rembrandt style
3. **Adjust Style Strength**: 
   - 30-50%: Subtle artistic touches
   - 50-80%: Balanced content + style
   - 80-100%: Maximum artistic transformation
4. **Generate**: Click "Generate Styled Portrait"
5. **Download**: Download your stylized image as PNG

## 💻 Programmatic Usage

### Basic Stylization

```python
from src.inference import StyleTransferInference

# Load model
model = StyleTransferInference.load_best_checkpoint()

# Stylize a selfie
styled = model.stylize_selfie(
    selfie_path="my_photo.jpg",
    artist="picasso",
    style_strength=0.8
)

# Save result
styled.save("output.png")
```

### Using Custom Style Image

```python
# Stylize with specific painting
styled = model.stylize_from_files(
    content_path="portrait.jpg",
    style_path="painting.jpg",
    artist="rembrandt",
    style_strength=1.0
)
```

### Batch Processing

```python
# Stylize multiple images
images = model.batch_stylize(
    content_paths=["photo1.jpg", "photo2.jpg", "photo3.jpg"],
    style_path="style.jpg",
    artist="picasso",
    batch_size=4
)

for i, img in enumerate(images):
    img.save(f"output_{i}.png")
```

## 📊 Evaluation Metrics

The system computes four key metrics:

| Metric | Purpose | Range | Better |
|--------|---------|-------|--------|
| **SSIM** | Content preservation (structural similarity) | 0-1 | Higher |
| **LPIPS** | Perceptual similarity to original | 0-1 | Lower |
| **Gram Distance** | Style match (Gram matrix distance) | 0+ | Lower |
| **Identity Similarity** | Face embedding similarity (FaceNet) | -1 to 1 | Higher |

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Image resolution
IMAGE_SIZE = 512

# Training hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Loss weights
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 10.0
IDENTITY_WEIGHT = 5.0
TV_WEIGHT = 1e-6

# Artists
ARTISTS = ["picasso", "rembrandt"]

# Paths
CHECKPOINTS_DIR = Path("checkpoints")
DATASETS_DIR = Path("datasets")
```

## 📈 Training Notes

- **GPU Requirements**: 1x Tesla V100 (32GB) or equivalent
- **Training Time**: ~30-50 hours for 50 epochs
- **Mixed Precision**: Enabled for faster training
- **Multi-GPU**: Supported via DataParallel

**Best Practices**:
1. Start with small learning rate (1e-4)
2. Monitor identity_similarity (should stay > 0.8)
3. Use validation set to detect overfitting
4. Save checkpoints regularly
5. Adjust loss weights based on metrics

## 🎯 Performance

Expected metrics on test set:
- **SSIM**: 0.75-0.85 (content preservation)
- **Identity Similarity**: 0.85-0.95 (identity preservation)
- **Gram Distance**: 0.01-0.05 (style matching)

## 🛠️ Troubleshooting

### "No module named 'facenet_pytorch'"
```bash
pip install facenet-pytorch
```

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in config.py
- Use `USE_AMP = True` for mixed precision

### "No faces detected"
- Ensure portrait is clear with visible face
- System falls back to center crop

### "Checkpoint not found"
- Train the model first: `python src/train.py`
- Or use untrained model for testing

## 📝 File Descriptions

### Core Models
- `encoder.py`: VGG19 feature extraction
- `adain.py`: Adaptive instance normalization
- `decoder.py`: Image reconstruction network
- `style_transfer.py`: Combined model architecture
- `identity.py`: FaceNet-based identity loss
- `losses.py`: All training loss functions

### Data Processing
- `style_preprocess.py`: Resize/crop paintings to 512x512
- `content_preprocess.py`: MTCNN face detection & cropping
- `data_loader.py`: PyTorch Dataset/DataLoader wrappers
- `metrics.py`: Evaluation metrics

### Training & Inference
- `train.py`: Full training loop with checkpointing
- `inference.py`: Inference wrapper for deployment
- `app.py`: Gradio web interface
- `config.py`: Centralized configuration

## 🎨 Example Workflow

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Preprocess (optional, if you add content faces)
python -m src.preprocess.style_preprocess
python -m src.preprocess.content_preprocess

# 3. Train (optional, skip to step 4 to use untrained model)
python src/train.py

# 4. Launch web app (uses best checkpoint if available)
python app.py

# 5. Open http://localhost:7860 and upload your selfie!
```

## 🤝 Contributing

For improvements:
1. Test on diverse portrait images
2. Experiment with loss weight adjustments
3. Try different decoder architectures
4. Optimize inference speed
5. Add more artist styles

## 📜 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- VGG19 implementation from torchvision
- FaceNet from facenet-pytorch
- MTCNN from facenet-pytorch
- AdaIN concept from Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
- Gradio framework for web interface

## 📞 Support

- Check `src/config.py` for all configurable parameters
- Review `src/train.py` for training customization
- See `src/inference.py` for programmatic API

---

**MuseAI MVP** - Transform portraits into art while preserving identity ✨
