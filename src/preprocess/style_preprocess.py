"""
Style Image Preprocessing
Processes Picasso and Rembrandt paintings for training.
"""
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import random

from PIL import Image
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    STYLE_RAW_DIR, STYLE_PROCESSED_DIR, STYLE_IMAGE_SIZE,
    ARTISTS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


class StylePreprocessor:
    """
    Preprocesses style images (paintings) for training.
    
    Steps:
    1. Load painting
    2. Convert to RGB
    3. Resize shortest side to 512 pixels
    4. Center-crop or pad to 512x512
    5. Save to data/style/{artist}/{train,val,test}
    """
    
    def __init__(
        self,
        source_dir: Path = STYLE_RAW_DIR,
        output_dir: Path = STYLE_PROCESSED_DIR,
        image_size: int = STYLE_IMAGE_SIZE,
        artists: List[str] = ARTISTS
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.artists = artists
        
        # Valid image extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def get_image_paths(self, artist: str) -> List[Path]:
        """Get all valid image paths for an artist."""
        artist_dir = self.source_dir / artist
        
        if not artist_dir.exists():
            print(f"Warning: Directory not found: {artist_dir}")
            return []
        
        images = []
        for ext in self.valid_extensions:
            images.extend(artist_dir.glob(f"*{ext}"))
            images.extend(artist_dir.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def process_image(self, image_path: Path) -> Optional[Image.Image]:
        """
        Process a single painting.
        
        Args:
            image_path: Path to the painting
        
        Returns:
            Processed PIL Image or None if failed
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB (handle RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize: make shortest side = target size
            w, h = img.size
            if w < h:
                new_w = self.image_size
                new_h = int(h * self.image_size / w)
            else:
                new_h = self.image_size
                new_w = int(w * self.image_size / h)
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center crop to square
            w, h = img.size
            left = (w - self.image_size) // 2
            top = (h - self.image_size) // 2
            right = left + self.image_size
            bottom = top + self.image_size
            
            img = img.crop((left, top, right, bottom))
            
            return img
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
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
    
    def process_artist(
        self,
        artist: str,
        save: bool = True
    ) -> dict:
        """
        Process all paintings for a single artist.
        
        Args:
            artist: Artist name ('picasso' or 'rembrandt')
            save: Whether to save processed images
        
        Returns:
            Statistics dictionary
        """
        print(f"\nProcessing {artist} paintings...")
        
        # Get image paths
        image_paths = self.get_image_paths(artist)
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            return {'artist': artist, 'total': 0, 'processed': 0}
        
        # Split dataset
        train_paths, val_paths, test_paths = self.split_dataset(image_paths)
        
        splits = {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        }
        
        stats = {
            'artist': artist,
            'total': len(image_paths),
            'train': len(train_paths),
            'val': len(val_paths),
            'test': len(test_paths),
            'processed': 0,
            'failed': 0
        }
        
        catalog = []
        
        for split_name, paths in splits.items():
            # Create output directory
            split_dir = self.output_dir / artist / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for path in tqdm(paths, desc=f"  {split_name}"):
                # Process image
                img = self.process_image(path)
                
                if img is not None:
                    if save:
                        # Save with consistent naming
                        output_path = split_dir / f"{path.stem}.jpg"
                        img.save(output_path, 'JPEG', quality=95)
                    
                    stats['processed'] += 1
                    
                    catalog.append({
                        'artist': artist,
                        'split': split_name,
                        'original_path': str(path),
                        'filename': f"{path.stem}.jpg",
                        'width': self.image_size,
                        'height': self.image_size
                    })
                else:
                    stats['failed'] += 1
        
        return stats, catalog
    
    def process_all(self, save: bool = True) -> pd.DataFrame:
        """
        Process all artists and create catalog.
        
        Returns:
            DataFrame with catalog of all processed images
        """
        print("=" * 50)
        print("Style Image Preprocessing")
        print("=" * 50)
        
        all_catalogs = []
        all_stats = []
        
        for artist in self.artists:
            stats, catalog = self.process_artist(artist, save=save)
            all_stats.append(stats)
            all_catalogs.extend(catalog)
        
        # Create catalog DataFrame
        catalog_df = pd.DataFrame(all_catalogs)
        
        # Save catalog
        if save:
            metadata_dir = self.output_dir.parent.parent / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            catalog_path = metadata_dir / "style_catalog.csv"
            catalog_df.to_csv(catalog_path, index=False)
            print(f"\nCatalog saved to: {catalog_path}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("Processing Summary")
        print("=" * 50)
        for stats in all_stats:
            print(f"\n{stats['artist'].upper()}:")
            print(f"  Total images: {stats['total']}")
            print(f"  Processed: {stats['processed']}")
            print(f"  Failed: {stats['failed']}")
            if 'train' in stats:
                print(f"  Train/Val/Test: {stats['train']}/{stats['val']}/{stats['test']}")
        
        return catalog_df


def main():
    """Run preprocessing."""
    preprocessor = StylePreprocessor()
    catalog = preprocessor.process_all(save=True)
    print(f"\nTotal processed images: {len(catalog)}")


if __name__ == "__main__":
    main()
