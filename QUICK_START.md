# MuseAI Quick Reference Guide

## 🚀 One-Minute Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
python app.py

# 3. Open http://localhost:7860
```

## 📸 How to Use

1. **Upload**: Click to upload portrait photo
2. **Select**: Choose Picasso or Rembrandt
3. **Adjust**: Set style strength (0-100%)
4. **Generate**: Click button
5. **Download**: Save PNG

## 💻 Quick Python Usage

```python
from src.inference import StyleTransferInference

# Load model
model = StyleTransferInference.load_best_checkpoint()

# Stylize
result = model.stylize_selfie(
    selfie_path="photo.jpg",
    artist="picasso",
    style_strength=0.8
)

# Save
result.save("styled.png")
```

## 🎯 System Requirements

- **GPU**: 1x V100 (32GB) or equivalent
- **CUDA**: 12.5+
- **Python**: 3.8+
- **RAM**: 8GB minimum
- **Disk**: 5GB for checkpoints

## 📊 Key Features

| Feature | Details |
|---------|---------|
| **Input** | JPG, PNG (any size) |
| **Output** | 512×512 PNG |
| **Artists** | Picasso, Rembrandt |
| **GPU** | Multi-GPU support |
| **Speed** | ~50-100ms per image |
| **Quality** | SSIM 0.75-0.85 |

## 🔧 Main Files

| File | Purpose |
|------|---------|
| `app.py` | Web interface (Gradio) |
| `src/train.py` | Training script |
| `src/inference.py` | Inference API |
| `src/config.py` | Configuration |
| `requirements.txt` | Dependencies |

## ⚙️ Configuration (src/config.py)

```python
# Training
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Loss weights
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 10.0
IDENTITY_WEIGHT = 5.0

# Image size
IMAGE_SIZE = 512
```

## 📝 Common Commands

```bash
# Launch web app
python app.py

# Preprocess paintings
python -m src.preprocess.style_preprocess

# Train model
python src/train.py

# Test encoder
python src/models/encoder.py

# Test dataset
python src/utils/data_loader.py
```

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA error | Reduce BATCH_SIZE |
| No faces detected | Use clear portrait |
| Module not found | `pip install -r requirements.txt` |
| Checkpoint not found | Checkpoint loads automatically if exists |

## 📈 Model Architecture

```
Input (512×512×3)
    ↓
VGG19 Encoder → features at relu4_1
    ↓
AdaIN (style transfer)
    ↓
Decoder → reconstruction
    ↓
Output (512×512×3)
    ↓
FaceNet (identity verification)
```

## 🎨 Style Strength Guide

- **0-30%**: Original photo with subtle artistic touches
- **30-60%**: Balanced content and style
- **60-90%**: Strong artistic transformation
- **90-100%**: Maximum style application

## 📂 Data Structure

```
datasets/
├── picasso/          ← 170 paintings
└── rembrandt/        ← 200 paintings

data/
└── style/
    ├── picasso/{train,val,test}/
    └── rembrandt/{train,val,test}/
```

## 🎓 Key Concepts

- **VGG19**: Frozen encoder for feature extraction
- **AdaIN**: Transfers style statistics to content
- **FaceNet**: Ensures facial identity preservation
- **Gram Matrix**: Represents image style
- **Instance Norm**: Adaptive normalization layer

## 📞 Need Help?

- Check README.md for detailed docs
- See individual module docstrings
- Review config.py for all settings
- Check logs/ for training metrics

---

**MuseAI MVP** - Transform portraits into art ✨
