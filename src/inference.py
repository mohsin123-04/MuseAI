"""
Inference Module for MuseAI
Stylize images without training. Simple API for deployment.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))

from src.config import (
    get_device, DEVICE, IMAGE_SIZE, ARTISTS, ARTIST_TO_IDX,
    CHECKPOINTS_DIR
)
from src.models.style_transfer import StyleTransferNetwork
from src.preprocess.content_preprocess import ContentPreprocessor


class StyleTransferInference:
    """
    Inference wrapper for style transfer.
    Handles loading models, preprocessing, and inference.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize inference model.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, use untrained model.
            device: Device to run on (default: cuda if available)
        """
        self.device = device or DEVICE
        
        # Load model
        self.model = StyleTransferNetwork(use_conditional=True)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print("Using untrained model (random initialization)")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing
        self.preprocessor = ContentPreprocessor(
            image_size=IMAGE_SIZE,
            use_face_detection=True
        )
        
        # For caching
        self._current_style = None
        self._current_artist = None
    
    def load_image(
        self,
        image_path: Union[str, Path]
    ) -> Image.Image:
        """Load image from file."""
        return Image.open(image_path).convert('RGB')
    
    def image_to_tensor(
        self,
        img: Image.Image,
        size: int = IMAGE_SIZE
    ) -> torch.Tensor:
        """
        Convert PIL image to normalized tensor.
        
        Args:
            img: PIL Image
            size: Target size
        
        Returns:
            Tensor [1, 3, H, W] in range [0, 1]
        """
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def tensor_to_image(
        self,
        tensor: torch.Tensor
    ) -> Image.Image:
        """
        Convert tensor back to PIL image.
        
        Args:
            tensor: Tensor [1, 3, H, W] in range [0, 1]
        
        Returns:
            PIL Image
        """
        tensor = tensor.squeeze(0).permute(1, 2, 0)
        tensor = (tensor * 255).clamp(0, 255).byte()
        return Image.fromarray(tensor.cpu().numpy())
    
    def stylize_from_files(
        self,
        content_path: Union[str, Path],
        style_path: Union[str, Path],
        artist: str = 'picasso',
        style_strength: float = 1.0,
        return_tensor: bool = False
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Stylize a content image using a style image.
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            artist: Artist name ('picasso' or 'rembrandt')
            style_strength: Style application strength (0.0-1.0)
            return_tensor: If True, return tensor instead of image
        
        Returns:
            Stylized image (PIL Image or torch.Tensor)
        """
        # Load images
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)
        
        # Convert to tensors
        content = self.image_to_tensor(content_img).to(self.device)
        style = self.image_to_tensor(style_img).to(self.device)
        
        # Get artist index
        artist_idx = torch.tensor([ARTIST_TO_IDX[artist]], device=self.device)
        
        # Stylize
        with torch.no_grad():
            stylized = self.model.stylize(
                content, style, artist_idx,
                style_strength=style_strength
            )
        
        if return_tensor:
            return stylized
        else:
            return self.tensor_to_image(stylized)
    
    def stylize_selfie(
        self,
        selfie_path: Union[str, Path],
        artist: str = 'picasso',
        style_strength: float = 1.0,
        style_image_path: Optional[Union[str, Path]] = None,
        return_tensor: bool = False
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Stylize a selfie/portrait image.
        
        This is the main user-facing method.
        
        Args:
            selfie_path: Path to selfie image
            artist: Artist to stylize in ('picasso' or 'rembrandt')
            style_strength: How much style to apply (0.0-1.0)
            style_image_path: Optional path to specific style image.
                             If None, uses a default for the artist.
            return_tensor: If True, return tensor
        
        Returns:
            Stylized image
        """
        # Load selfie
        selfie = self.load_image(selfie_path)
        
        # Preprocess face (detect/crop)
        processed_selfie, face_detected = self.preprocessor.process_image(Path(selfie_path))
        
        if processed_selfie is None:
            raise ValueError(f"Could not process image: {selfie_path}")
        
        print(f"Face detected: {face_detected}")
        
        # If no specific style image provided, try to find one
        if style_image_path is None:
            # Try to find a style image from datasets
            style_dir = Path(f"datasets/{artist}")
            if not style_dir.exists():
                print(f"Warning: No style images found in {style_dir}")
                # Create a random style tensor as fallback
                style_tensor = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
            else:
                # Find first image in directory
                from src.utils.data_loader import RawStyleDataset
                style_dataset = RawStyleDataset(artist=artist)
                if len(style_dataset) > 0:
                    style_tensor = style_dataset[0]['image'].unsqueeze(0).to(self.device)
                    print(f"Using style from: {style_dataset[len(style_dataset)//2]['path']}")
                else:
                    print("Warning: No style images found")
                    style_tensor = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
        else:
            style_img = self.load_image(style_image_path)
            style_tensor = self.image_to_tensor(style_img).to(self.device)
        
        # Convert selfie to tensor
        content_tensor = self.image_to_tensor(processed_selfie).to(self.device)
        
        # Get artist index
        artist_idx = torch.tensor([ARTIST_TO_IDX[artist]], device=self.device)
        
        # Stylize
        with torch.no_grad():
            stylized = self.model.stylize(
                content_tensor, style_tensor, artist_idx,
                style_strength=style_strength
            )
        
        if return_tensor:
            return stylized
        else:
            return self.tensor_to_image(stylized)
    
    def batch_stylize(
        self,
        content_paths: list,
        style_path: Union[str, Path],
        artist: str = 'picasso',
        style_strength: float = 1.0,
        batch_size: int = 4
    ) -> list:
        """
        Stylize multiple images with the same style.
        
        Args:
            content_paths: List of content image paths
            style_path: Path to style image
            artist: Artist name
            style_strength: Style strength
            batch_size: Batch size for processing
        
        Returns:
            List of stylized images
        """
        style_img = self.load_image(style_path)
        style_tensor = self.image_to_tensor(style_img).to(self.device)
        artist_idx = torch.tensor([ARTIST_TO_IDX[artist]], device=self.device)
        
        results = []
        
        for i in range(0, len(content_paths), batch_size):
            batch_paths = content_paths[i:i+batch_size]
            
            # Load batch
            batch_tensors = []
            for path in batch_paths:
                img = self.load_image(path)
                tensor = self.image_to_tensor(img)
                batch_tensors.append(tensor)
            
            content_batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Expand style to batch size
            style_batch = style_tensor.repeat(content_batch.size(0), 1, 1, 1)
            artist_batch = artist_idx.repeat(content_batch.size(0))
            
            # Stylize
            with torch.no_grad():
                stylized_batch = self.model.stylize(
                    content_batch, style_batch, artist_batch,
                    style_strength=style_strength
                )
            
            # Convert back to images
            for j in range(stylized_batch.size(0)):
                img = self.tensor_to_image(stylized_batch[j:j+1])
                results.append(img)
        
        return results
    
    @staticmethod
    def list_artists() -> list:
        """Get list of available artists."""
        return ARTISTS
    
    @staticmethod
    def load_best_checkpoint() -> 'StyleTransferInference':
        """
        Load the best checkpoint if available.
        Falls back to untrained model if not found.
        """
        best_checkpoint = CHECKPOINTS_DIR / "checkpoint_best.pt"
        
        if best_checkpoint.exists():
            print(f"Loading best checkpoint: {best_checkpoint}")
            return StyleTransferInference(checkpoint_path=str(best_checkpoint))
        else:
            print("No checkpoint found, using untrained model")
            return StyleTransferInference()


def main():
    """Test inference."""
    print("MuseAI Inference Test")
    print("=" * 60)
    
    # Load model
    model = StyleTransferInference.load_best_checkpoint()
    
    print(f"\nAvailable artists: {model.list_artists()}")
    
    print("\nInference module ready for deployment!")


if __name__ == "__main__":
    main()
