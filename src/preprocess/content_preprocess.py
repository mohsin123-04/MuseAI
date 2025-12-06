"""
Content Image Preprocessing
Processes face images (selfies, portraits) for training.
Uses MTCNN for face detection with fallback to center crop.
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import random

from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    CONTENT_RAW_DIR, CONTENT_PROCESSED_DIR, CONTENT_IMAGE_SIZE,
    FACE_DETECTION_THRESHOLD, FACE_BBOX_EXPAND_RATIO, MIN_FACE_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


class ContentPreprocessor:
    """
    Preprocesses content images (faces/portraits) for training.
    
    Steps:
    1. Detect face using MTCNN
    2. Expand bounding box by 1.4x
    3. Square crop around head
    4. Resize to 512x512
    5. If detection fails, fallback to center crop
    """
    
    def __init__(
        self,
        source_dir: Path = CONTENT_RAW_DIR,
        output_dir: Path = CONTENT_PROCESSED_DIR,
        image_size: int = CONTENT_IMAGE_SIZE,
        use_face_detection: bool = True
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.use_face_detection = use_face_detection
        
        # Valid image extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # Initialize face detector
        self.mtcnn = None
        if use_face_detection:
            self._init_face_detector()
    
    def _init_face_detector(self):
        """Initialize MTCNN face detector."""
        try:
            from facenet_pytorch import MTCNN
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mtcnn = MTCNN(
                image_size=self.image_size,
                margin=0,
                min_face_size=MIN_FACE_SIZE,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=device,
                keep_all=False
            )
            print(f"MTCNN initialized on {device}")
        except ImportError:
            print("Warning: facenet_pytorch not installed. Using center crop.")
            self.mtcnn = None
    
    def get_image_paths(self, subdir: str = None) -> List[Path]:
        """Get all valid image paths."""
        search_dir = self.source_dir / subdir if subdir else self.source_dir
        
        if not search_dir.exists():
            print(f"Warning: Directory not found: {search_dir}")
            return []
        
        images = []
        for ext in self.valid_extensions:
            images.extend(search_dir.glob(f"**/*{ext}"))
            images.extend(search_dir.glob(f"**/*{ext.upper()}"))
        
        return sorted(images)
    
    def detect_face(self, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face and return bounding box.
        
        Args:
            img: PIL Image
        
        Returns:
            Bounding box (x1, y1, x2, y2) or None if no face detected
        """
        if self.mtcnn is None:
            return None
        
        try:
            import torch
            
            # Detect face
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None or len(boxes) == 0:
                return None
            
            # Get highest confidence detection
            if probs is not None:
                best_idx = probs.argmax()
                if probs[best_idx] < FACE_DETECTION_THRESHOLD:
                    return None
                box = boxes[best_idx]
            else:
                box = boxes[0]
            
            return tuple(map(int, box))
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return None
    
    def expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        img_size: Tuple[int, int],
        expand_ratio: float = FACE_BBOX_EXPAND_RATIO
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box and make it square.
        
        Args:
            bbox: Original bounding box (x1, y1, x2, y2)
            img_size: Image dimensions (width, height)
            expand_ratio: Expansion factor (1.4 = 40% larger)
        
        Returns:
            Expanded square bounding box
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        img_w, img_h = img_size
        
        # Calculate center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Expand and make square
        size = int(max(w, h) * expand_ratio)
        
        # Calculate new bounds
        new_x1 = max(0, cx - size // 2)
        new_y1 = max(0, cy - size // 2)
        new_x2 = min(img_w, cx + size // 2)
        new_y2 = min(img_h, cy + size // 2)
        
        # Ensure square (might not be exact at edges)
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        
        if new_w != new_h:
            min_dim = min(new_w, new_h)
            new_x2 = new_x1 + min_dim
            new_y2 = new_y1 + min_dim
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    def center_crop(self, img: Image.Image) -> Image.Image:
        """Fallback: center crop to square."""
        w, h = img.size
        size = min(w, h)
        
        left = (w - size) // 2
        top = (h - size) // 2
        right = left + size
        bottom = top + size
        
        return img.crop((left, top, right, bottom))
    
    def process_image(
        self,
        image_path: Path,
        use_detection: bool = True
    ) -> Tuple[Optional[Image.Image], bool]:
        """
        Process a single content image.
        
        Args:
            image_path: Path to the image
            use_detection: Whether to use face detection
        
        Returns:
            Tuple of (processed image, whether face was detected)
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            face_detected = False
            
            if use_detection and self.mtcnn is not None:
                # Try face detection
                bbox = self.detect_face(img)
                
                if bbox is not None:
                    # Expand bounding box
                    expanded_bbox = self.expand_bbox(bbox, img.size)
                    
                    # Crop to face region
                    img = img.crop(expanded_bbox)
                    face_detected = True
            
            if not face_detected:
                # Fallback to center crop
                img = self.center_crop(img)
            
            # Resize to target size
            img = img.resize(
                (self.image_size, self.image_size),
                Image.Resampling.LANCZOS
            )
            
            return img, face_detected
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, False
    
    def split_dataset(
        self,
        image_paths: List[Path],
        seed: int = 42
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split images into train/val/test sets."""
        random.seed(seed)
        shuffled = image_paths.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]
        
        return train, val, test
    
    def process_all(self, save: bool = True) -> pd.DataFrame:
        """
        Process all content images.
        
        Returns:
            DataFrame with catalog of all processed images
        """
        print("=" * 50)
        print("Content Image Preprocessing")
        print("=" * 50)
        
        # Get image paths
        image_paths = self.get_image_paths()
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("No images found to process.")
            return pd.DataFrame()
        
        # Split dataset
        train_paths, val_paths, test_paths = self.split_dataset(image_paths)
        
        splits = {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        }
        
        stats = {
            'total': len(image_paths),
            'processed': 0,
            'face_detected': 0,
            'center_cropped': 0,
            'failed': 0
        }
        
        catalog = []
        
        for split_name, paths in splits.items():
            # Create output directory
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for path in tqdm(paths, desc=f"Processing {split_name}"):
                # Process image
                img, face_detected = self.process_image(path)
                
                if img is not None:
                    if save:
                        output_path = split_dir / f"{path.stem}.jpg"
                        img.save(output_path, 'JPEG', quality=95)
                    
                    stats['processed'] += 1
                    if face_detected:
                        stats['face_detected'] += 1
                    else:
                        stats['center_cropped'] += 1
                    
                    catalog.append({
                        'split': split_name,
                        'original_path': str(path),
                        'filename': f"{path.stem}.jpg",
                        'width': self.image_size,
                        'height': self.image_size,
                        'face_detected': face_detected
                    })
                else:
                    stats['failed'] += 1
        
        # Create catalog DataFrame
        catalog_df = pd.DataFrame(catalog)
        
        # Save catalog
        if save and len(catalog) > 0:
            metadata_dir = self.output_dir.parent.parent / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            catalog_path = metadata_dir / "content_catalog.csv"
            catalog_df.to_csv(catalog_path, index=False)
            print(f"\nCatalog saved to: {catalog_path}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("Processing Summary")
        print("=" * 50)
        print(f"Total images: {stats['total']}")
        print(f"Processed: {stats['processed']}")
        print(f"Face detected: {stats['face_detected']}")
        print(f"Center cropped (fallback): {stats['center_cropped']}")
        print(f"Failed: {stats['failed']}")
        
        return catalog_df


def process_single_image(
    image_path: str,
    output_path: str = None,
    image_size: int = 512
) -> Optional[Image.Image]:
    """
    Convenience function to process a single image.
    Useful for inference.
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save processed image
        image_size: Target size
    
    Returns:
        Processed PIL Image
    """
    preprocessor = ContentPreprocessor(
        output_dir=Path("."),
        image_size=image_size,
        use_face_detection=True
    )
    
    img, face_detected = preprocessor.process_image(Path(image_path))
    
    if img is not None and output_path:
        img.save(output_path, 'JPEG', quality=95)
        print(f"Saved to: {output_path}")
        print(f"Face detected: {face_detected}")
    
    return img


def main():
    """Run preprocessing."""
    preprocessor = ContentPreprocessor()
    catalog = preprocessor.process_all(save=True)
    print(f"\nTotal processed images: {len(catalog)}")


if __name__ == "__main__":
    main()
