"""
Data Loading Utilities for MuseAI
PyTorch datasets and dataloaders for style transfer training.
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    STYLE_PROCESSED_DIR, CONTENT_PROCESSED_DIR, DATASETS_DIR,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    ARTISTS, ARTIST_TO_IDX,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
)


class StyleContentDataset(Dataset):
    """
    Dataset that provides paired content and style images for training.
    
    Each sample contains:
    - content: A face image
    - style: A painting from the selected artist
    - artist_idx: Index of the artist (0=Picasso, 1=Rembrandt)
    """
    
    def __init__(
        self,
        content_dir: Path = CONTENT_PROCESSED_DIR,
        style_dir: Path = STYLE_PROCESSED_DIR,
        split: str = 'train',
        image_size: int = IMAGE_SIZE,
        augment: bool = True
    ):
        self.content_dir = Path(content_dir) / split
        self.style_dir = Path(style_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Get content images
        self.content_images = self._get_images(self.content_dir)
        
        # Get style images per artist
        self.style_images = {}
        for artist in ARTISTS:
            artist_dir = self.style_dir / artist / split
            self.style_images[artist] = self._get_images(artist_dir)
        
        # Build transforms
        self.content_transform = self._build_content_transform()
        self.style_transform = self._build_style_transform()
        
        print(f"Dataset '{split}' initialized:")
        print(f"  Content images: {len(self.content_images)}")
        for artist, images in self.style_images.items():
            print(f"  {artist.capitalize()} styles: {len(images)}")
    
    def _get_images(self, directory: Path) -> List[Path]:
        """Get all image paths from a directory."""
        if not directory.exists():
            return []
        
        extensions = {'.jpg', '.jpeg', '.png'}
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
        
        return sorted(images)
    
    def _build_content_transform(self) -> transforms.Compose:
        """Build transforms for content images."""
        transform_list = []
        
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
            ])
        
        transform_list.extend([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        return transforms.Compose(transform_list)
    
    def _build_style_transform(self) -> transforms.Compose:
        """Build transforms for style images."""
        transform_list = []
        
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ])
        
        transform_list.extend([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.content_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
            - content: Content image tensor [3, H, W]
            - style: Style image tensor [3, H, W]
            - artist_idx: Artist index tensor
            - artist_name: Artist name string
        """
        # Load content image
        content_path = self.content_images[idx]
        content_img = Image.open(content_path).convert('RGB')
        content = self.content_transform(content_img)
        
        # Randomly select artist
        artist = random.choice(ARTISTS)
        artist_idx = ARTIST_TO_IDX[artist]
        
        # Randomly select style image from that artist
        if len(self.style_images[artist]) > 0:
            style_path = random.choice(self.style_images[artist])
            style_img = Image.open(style_path).convert('RGB')
            style = self.style_transform(style_img)
        else:
            # Fallback: random noise (shouldn't happen in practice)
            style = torch.rand(3, self.image_size, self.image_size)
        
        return {
            'content': content,
            'style': style,
            'artist_idx': torch.tensor(artist_idx, dtype=torch.long),
            'artist_name': artist,
            'content_path': str(content_path),
            'style_path': str(style_path) if len(self.style_images[artist]) > 0 else ''
        }


class StyleDataset(Dataset):
    """Dataset for loading only style images (useful for evaluation)."""
    
    def __init__(
        self,
        style_dir: Path = STYLE_PROCESSED_DIR,
        artist: str = None,
        split: str = 'train',
        image_size: int = IMAGE_SIZE
    ):
        self.style_dir = Path(style_dir)
        self.image_size = image_size
        
        # Get images
        self.images = []
        self.artist_labels = []
        
        artists_to_load = [artist] if artist else ARTISTS
        
        for art in artists_to_load:
            art_dir = self.style_dir / art / split
            art_images = self._get_images(art_dir)
            self.images.extend(art_images)
            self.artist_labels.extend([ARTIST_TO_IDX[art]] * len(art_images))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def _get_images(self, directory: Path) -> List[Path]:
        """Get all image paths from a directory."""
        if not directory.exists():
            return []
        
        extensions = {'.jpg', '.jpeg', '.png'}
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
        
        return sorted(images)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return {
            'image': img,
            'artist_idx': torch.tensor(self.artist_labels[idx], dtype=torch.long),
            'path': str(img_path)
        }


class RawStyleDataset(Dataset):
    """
    Dataset for loading raw (unprocessed) style images.
    Useful when you want to skip preprocessing or use original dataset structure.
    """
    
    def __init__(
        self,
        datasets_dir: Path = DATASETS_DIR,
        artist: str = None,
        image_size: int = IMAGE_SIZE
    ):
        self.datasets_dir = Path(datasets_dir)
        self.image_size = image_size
        
        # Get images
        self.images = []
        self.artist_labels = []
        
        artists_to_load = [artist] if artist else ARTISTS
        
        for art in artists_to_load:
            art_dir = self.datasets_dir / art
            art_images = self._get_images(art_dir)
            self.images.extend(art_images)
            self.artist_labels.extend([ARTIST_TO_IDX[art]] * len(art_images))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f"RawStyleDataset: Found {len(self.images)} images")
    
    def _get_images(self, directory: Path) -> List[Path]:
        """Get all image paths from a directory."""
        if not directory.exists():
            print(f"Warning: Directory not found: {directory}")
            return []
        
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"**/*{ext}"))
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        
        return sorted(images)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.images[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = torch.zeros(3, self.image_size, self.image_size)
        
        return {
            'image': img,
            'artist_idx': torch.tensor(self.artist_labels[idx], dtype=torch.long),
            'path': str(img_path)
        }


def create_dataloaders(
    content_dir: Path = CONTENT_PROCESSED_DIR,
    style_dir: Path = STYLE_PROCESSED_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY
) -> Dict[str, DataLoader]:
    """
    Create train, val, and test dataloaders.
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = StyleContentDataset(
            content_dir=content_dir,
            style_dir=style_dir,
            split=split,
            augment=(split == 'train')
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')
        )
    
    return dataloaders


def create_style_dataloader(
    style_dir: Path = STYLE_PROCESSED_DIR,
    artist: str = None,
    split: str = 'train',
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True
) -> DataLoader:
    """Create a dataloader for style images only."""
    dataset = StyleDataset(
        style_dir=style_dir,
        artist=artist,
        split=split
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )


if __name__ == "__main__":
    # Test dataset loading
    print("Testing StyleContentDataset...")
    
    # Test with raw dataset directory
    raw_dataset = RawStyleDataset()
    print(f"\nRaw dataset size: {len(raw_dataset)}")
    
    if len(raw_dataset) > 0:
        sample = raw_dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample artist: {sample['artist_idx']}")
    
    # Try to create full dataset (might fail if content not prepared)
    try:
        dataset = StyleContentDataset(split='train')
        print(f"\nFull dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Content shape: {sample['content'].shape}")
            print(f"Style shape: {sample['style'].shape}")
            print(f"Artist: {sample['artist_name']}")
    except Exception as e:
        print(f"\nCould not create full dataset (expected if content not prepared): {e}")
