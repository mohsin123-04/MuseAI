# MuseAI Terminal CLI User Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the CLI
python cli.py

# 3. Follow the menu
```

The application is now ready for use.

---

## Main Menu Options

```
MuseAI - Neural Style Transfer (Terminal Version)
============================================================

1. Create New User / Select User
2. Upload Image
3. Apply Style Transfer
4. List Users and Images
5. View Output Images
6. Delete User Data
7. Help and Commands
0. Exit
```

---

## Typical Workflow

### First Time Users

```
Step 1: Select option 1 - "Create New User"
        Input username: john_doe
        User created

Step 2: Select option 2 - "Upload Image"
        Paste image path: C:\Users\Photos\selfie.jpg
        Image uploaded

Step 3: Select option 3 - "Apply Style Transfer"
        Choose image: selfie.jpg
        Choose artist: 1 (Picasso) or 2 (Rembrandt)
        Style strength: 1.0
        Processing...
        Output saved

Step 4: Select option 5 - "View Output Images"
        Output folder displayed
```

### Returning Users

```
Step 1: Select option 1 - Choose existing user "john_doe"
Step 2: Select option 2 - Upload new image or reuse existing
Step 3: Select option 3 - Try different artist or style strength
```

---

## Artistic Styles

### Picasso
- Style: Cubism, abstract geometric shapes
- Best for: Creative, artistic transformation
- Use case: Bold artistic visual effect

### Rembrandt
- Style: Classical oil painting, golden lighting
- Best for: Realistic portrait enhancement
- Use case: Refined, classical appearance

### Style Strength

| Strength | Effect | Use Case |
|----------|--------|----------|
| 0.0-0.3 | Subtle artistic touches | Preserve original image |
| 0.3-0.6 | Balanced blend | Recommended for most cases |
| 0.6-0.9 | Strong artistic | Noticeable transformation |
| 0.9-1.0 | Maximum style | Full artistic effect |

---

## Folder Structure

Data is organized by user:

```
uploads/
├── user_1/
│   ├── input/
│   │   ├── selfie.jpg
│   │   ├── portrait.jpg
│   │   └── ...
│   └── output/
│       ├── picasso_20250605_143022.png
│       ├── picasso_20250605_143522.png
│       ├── rembrandt_20250605_143547.png
│       └── ...
├── user_2/
├── john_doe/
└── ...
```

Key points:
- Each user has a dedicated folder
- input/: Directory for uploaded images
- output/: Directory for processed results
- Results are organized by date, time, and artist

---

## Uploading Images

### Method 1: Copy Image File

```
Menu → 2. Upload Image
     → 1. Copy image file
     → Paste full path: C:\Users\John\Pictures\selfie.jpg
     Image copied to user's input folder
```

Supported formats:
- JPG / JPEG
- PNG
- BMP

### Method 2: Manual Copy

Copy images directly using command line:

```powershell
# Copy to user's input folder
copy "C:\Users\John\Pictures\selfie.jpg" "uploads\john_doe\input\"
```

---

## Running Style Transfer

### Complete Workflow

```
Menu → 3. Apply Style Transfer

Multiple images found:
   1. selfie.jpg
   2. portrait.jpg
   Select image number: 1
   
STYLE TRANSFER
   Input image: selfie.jpg
   
   Available artists:
     1. Picasso
     2. Rembrandt
   
   Select artist (1 or 2): 1
   Style strength (0-1.0, default 1.0): 0.8
   
   Processing: Picasso style (strength: 0.8)
   
   Style transfer complete
      Output: uploads\john_doe\output\picasso_20250605_143022.png
      Size: 2.45 MB
```

---

## Viewing Results

### Option 5: View Output Images

```
Menu → 5. View Output Images

OUTPUT IMAGES
────────────────────────────────
1. picasso_20250605_143022.png
   Size: 2.45 MB | Created: 2025-06-05 14:30:22

2. rembrandt_20250605_143547.png
   Size: 2.43 MB | Created: 2025-06-05 14:35:47

Output folder: C:\MuseAI\uploads\john_doe\output
```

### Manual Navigation

```powershell
# Open output folder in File Explorer
explorer "uploads\john_doe\output"

# List all output images
ls uploads\john_doe\output
```

---

## Managing Users

### Create New User

```
Menu → 1. Create New User / Select User
     → Create New User
     → Enter user ID: sarah_photos
     User set to: sarah_photos
```

### Switch Between Users

```
Menu → 1. Create New User / Select User
     
Existing Users:
   1. john_doe
   2. sarah_photos
   3. Create New User
     
Select user number: 1
User set to: john_doe
```

### List All Users and Images

```
Menu → 4. List Users and Images

USERS & IMAGES
────────────────────────────────
User: john_doe
   Input (2):
      • selfie.jpg (245.3 KB)
      • portrait.jpg (512.7 KB)
   Output (2):
      • picasso_20250605_143022.png (2.45 MB)
      • rembrandt_20250605_143547.png (2.43 MB)

User: sarah_photos
   Input (1):
      • photo.jpg (389.1 KB)
   Output: (empty)
```

### Delete User Data

```
Menu → 6. Delete User Data

DELETE USER DATA
────────────────────────────────
Existing users:
   1. john_doe
   2. sarah_photos

Select user to delete (0 to cancel): 1
Delete all data for 'john_doe'? (yes/no): yes
User 'john_doe' deleted
```

---

## Advanced Usage

### Custom Upload Directory

```bash
# Use a different folder for uploads
python cli.py --upload-dir "C:\MyPhotos\StyleTransfer"

# Create custom folder structure
python cli.py --upload-dir "D:\Projects\MuseAI\uploads"
```

### Command Line Help

```bash
python cli.py --help

usage: cli.py [-h] [--upload-dir UPLOAD_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --upload-dir UPLOAD_DIR
                        Base directory for user uploads (default: uploads/)
```

---

## Troubleshooting

### Model Loading Error

Cause: Checkpoint file does not exist  
Solution:
```bash
# Verify checkpoint exists
ls checkpoints/

# If missing, train a model first
python src/train.py
```

### No Faces Detected

Cause: Image does not contain a clear facial region  
Solution:
- Use clear portrait photography with visible face
- Ensure face occupies at least 30% of image area
- Try alternative image
- Center crop image before uploading

### Processing Performance

Image processing requires computational time:
- GPU performs neural network calculations
- CUDA processes style transfer algorithm
- Output image generation

To optimize performance:
- Use high-performance GPU (Tesla V100 preferred)
- Ensure GPU has sufficient VRAM (8GB minimum)
- Reduce image size by editing config.IMAGE_SIZE

### File Path Not Recognized

Cause: Incorrect path format  
Solution:
```bash
# Use full absolute paths
C:\Users\John\Pictures\selfie.jpg  # Correct

# Do not use relative paths
Pictures\selfie.jpg  # Incorrect

# Copy full path from File Explorer properties
```

### Permission Denied

Cause: File is locked or insufficient folder permissions  
Solution:
- Close image file in other applications
- Verify folder permissions
- Run terminal with administrator privileges

---

## Performance Specifications

| Metric | Value |
|--------|-------|
| Processing time per image | 30-60 seconds |
| Output file size | 2-3 MB |
| Output resolution | 512×512 pixels |
| GPU memory requirement | 6-8 GB |
| CPU utilization | 20-30% |

---

## Example Sessions

### Session 1: First Time Use

```bash
$ python cli.py

MuseAI - Neural Style Transfer (Terminal Version)
============================================================
Loading MuseAI model...
Model loaded successfully!
Using device: CUDA (GPU 0: Tesla V100)
Uploads folder: C:\MuseAI\uploads

MAIN MENU
─────────────────────────────
1. Create New User / Select User
2. Upload Image
3. Apply Style Transfer
...

Enter choice (0-7): 1

USER SELECTION
─────────────────────────────
No existing users found. Create a new user.

Enter user ID (alphanumeric, no spaces): john_doe

User set to: john_doe
   Input folder:  C:\MuseAI\uploads\john_doe\input
   Output folder: C:\MuseAI\uploads\john_doe\output

MAIN MENU
─────────────────────────────

Enter choice (0-7): 2

UPLOAD IMAGE
─────────────────────────────
Input folder: C:\MuseAI\uploads\john_doe\input

Options:
1. Copy image file
2. Paste from clipboard path
3. Cancel

Choice: 1

Enter full path to image: C:\Users\John\Pictures\selfie.jpg

Copying: selfie.jpg
Image uploaded: C:\MuseAI\uploads\john_doe\input\selfie.jpg
   File size: 1.25 MB

MAIN MENU
─────────────────────────────

Enter choice (0-7): 3

STYLE TRANSFER
─────────────────────────────
Input image: selfie.jpg

Available artists:
  1. Picasso
  2. Rembrandt

Select artist (1 or 2): 1
Style strength (0-1.0, default 1.0): 0.8

Processing: Picasso style (strength: 0.8)

Style transfer complete
   Output: C:\MuseAI\uploads\john_doe\output\picasso_20250605_143022.png
   Size: 2.45 MB
```

### Session 2: Apply Different Style

```bash
Enter choice (0-7): 3

STYLE TRANSFER
Input image: selfie.jpg
Available artists:
  1. Picasso
  2. Rembrandt

Select artist (1 or 2): 2
Style strength (0-1.0, default 1.0): 0.6

Processing: Rembrandt style (strength: 0.6)

Style transfer complete
   Output: C:\MuseAI\uploads\john_doe\output\rembrandt_20250605_143547.png
   Size: 2.43 MB
```

---

## File Locations

| File/Folder | Location |
|-------------|----------|
| CLI Script | cli.py |
| User Data | uploads/ |
| Checkpoints | checkpoints/ |
| Logs | logs/ |
| Configuration | src/config.py |

---

## Production Usage

### Multi-User Processing

```bash
# Terminal 1: User 1
python cli.py --upload-dir "uploads_user1"

# Terminal 2: User 2
python cli.py --upload-dir "uploads_user2"

# Both run independently
```

### Batch Processing

```python
# Batch processing script
from src.inference import StyleTransferInference
import os
from pathlib import Path

model = StyleTransferInference.load_best_checkpoint()

for image in Path("uploads/john_doe/input").glob("*.jpg"):
    for artist in ["picasso", "rembrandt"]:
        result = model.stylize_selfie(
            selfie_path=str(image),
            artist=artist,
            style_strength=0.8
        )
        output_name = f"{artist}_{image.stem}.png"
        result.save(f"uploads/john_doe/output/{output_name}")
        print(f"Processed: {output_name}")
```

---

## Usage Tips

### Tip 1: Style Strength Variation
```
Same image, different configurations:
├─ Picasso (0.5) = Light cubism
├─ Picasso (1.0) = Bold cubism
├─ Rembrandt (0.5) = Subtle oil painting
└─ Rembrandt (1.0) = Strong classical style
```

### Tip 2: Multiple Portrait Variations
```
Use different source images:
├─ Passport photo → Formal Rembrandt
├─ Action photo → Dynamic Picasso
├─ Close-up → Detailed style
└─ Group photo → Interesting results
```

### Tip 3: Organized File Management
```
uploads/
├── john_doe/
│   ├── input/
│   │   ├── selfie_001.jpg
│   │   ├── selfie_002.jpg
│   │   └── selfie_003.jpg
│   └── output/  (all results)
```

---

## Support and Documentation

Reference materials:
1. This guide: CLI_GUIDE.md
2. Built-in help: Run option 7 in menu
3. Architecture details: README.md
4. Code reference: src/inference.py

Frequently asked questions:

Q: Is GPU acceleration available?  
A: Yes. Automatic GPU detection with CUDA 12.5+

Q: What is the expected processing time?  
A: Approximately 30-60 seconds per image depending on GPU

Q: Can multiple images be processed?  
A: Yes. Upload multiple images and process each individually

Q: Can output quality be adjusted?  
A: Yes. Edit src/config.py and retrain the model

Q: Where are processed images saved?  
A: uploads/<username>/output/

---

## Getting Started

1. Execute: python cli.py
2. Create user profile: Option 1
3. Upload image: Option 2
4. Apply style transfer: Option 3
5. View results: Option 5
6. Apply alternative styles: Repeat option 3
7. Export results: Copy from output folder

---

**MuseAI Terminal CLI** - Neural Style Transfer Application
