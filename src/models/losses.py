"""
Loss Functions for MuseAI Style Transfer
Complete loss computation including content, style, identity, and TV losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    CONTENT_WEIGHT, STYLE_WEIGHT, IDENTITY_WEIGHT, TV_WEIGHT,
    STYLE_LAYERS, CONTENT_LAYERS
)
from src.models.encoder import VGGEncoder
from src.models.identity import FaceNetIdentity


class GramMatrix(nn.Module):
    """Compute Gram matrix for style representation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix.
        
        Args:
            x: Feature tensor [B, C, H, W]
        
        Returns:
            Gram matrix [B, C, C]
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)


class ContentLoss(nn.Module):
    """Content loss using VGG features."""
    
    def __init__(self, layer: str = 'relu4_1'):
        super(ContentLoss, self).__init__()
        self.layer = layer
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        generated_features: Dict[str, torch.Tensor],
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute content loss.
        
        For style transfer, the target is typically the AdaIN output features,
        not the original content features. This trains the decoder to 
        faithfully reconstruct from AdaIN-transformed features.
        """
        return self.mse(generated_features[self.layer], target_features)


class StyleLoss(nn.Module):
    """Style loss using Gram matrices at multiple VGG layers."""
    
    def __init__(self, layers: list = None):
        super(StyleLoss, self).__init__()
        self.layers = layers or STYLE_LAYERS
        self.gram = GramMatrix()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        generated_features: Dict[str, torch.Tensor],
        style_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute style loss across multiple layers."""
        loss = 0.0
        for layer in self.layers:
            gen_gram = self.gram(generated_features[layer])
            style_gram = self.gram(style_features[layer])
            loss += self.mse(gen_gram, style_gram)
        return loss / len(self.layers)


class TotalVariationLoss(nn.Module):
    """Total variation loss for spatial smoothness."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute TV loss.
        
        Encourages spatial smoothness by penalizing high-frequency variations.
        """
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return (diff_h.mean() + diff_w.mean()) / 2


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    Alternative to Gram-based style loss.
    """
    
    def __init__(self, layers: list = None):
        super(PerceptualLoss, self).__init__()
        self.layers = layers or ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        generated_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute perceptual loss as feature-space L2 distance."""
        loss = 0.0
        for layer in self.layers:
            loss += self.mse(generated_features[layer], target_features[layer])
        return loss / len(self.layers)


class CombinedLoss(nn.Module):
    """
    Combined loss function for MuseAI style transfer training.
    
    Total Loss = λ_content * L_content + λ_style * L_style + 
                 λ_identity * L_identity + λ_tv * L_tv
    """
    
    def __init__(
        self,
        content_weight: float = CONTENT_WEIGHT,
        style_weight: float = STYLE_WEIGHT,
        identity_weight: float = IDENTITY_WEIGHT,
        tv_weight: float = TV_WEIGHT,
        use_identity_loss: bool = True
    ):
        super(CombinedLoss, self).__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.identity_weight = identity_weight
        self.tv_weight = tv_weight
        self.use_identity_loss = use_identity_loss
        
        # Loss components
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.tv_loss = TotalVariationLoss()
        
        # VGG encoder for feature extraction
        self.encoder = VGGEncoder(requires_grad=False)
        
        # Identity loss (FaceNet)
        if use_identity_loss:
            self.identity_loss = FaceNetIdentity()
        else:
            self.identity_loss = None
    
    def forward(
        self,
        generated: torch.Tensor,
        content: torch.Tensor,
        style: torch.Tensor,
        adain_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            generated: Generated/stylized image [B, 3, H, W]
            content: Original content image [B, 3, H, W]
            style: Style reference image [B, 3, H, W]
            adain_features: Features after AdaIN [B, 512, H, W]
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss values for logging
        """
        # Get VGG features
        gen_features = self.encoder(generated, return_all=True)
        style_features = self.encoder(style, return_all=True)
        
        # Content loss: decoder should reconstruct AdaIN features
        c_loss = self.content_loss(gen_features, adain_features)
        
        # Style loss: match Gram matrices
        s_loss = self.style_loss(gen_features, style_features)
        
        # Total variation loss
        tv = self.tv_loss(generated)
        
        # Identity loss (optional)
        if self.use_identity_loss and self.identity_loss is not None:
            id_loss, id_similarity = self.identity_loss(content, generated)
        else:
            id_loss = torch.tensor(0.0, device=generated.device)
            id_similarity = torch.tensor(0.0, device=generated.device)
        
        # Combine losses
        total_loss = (
            self.content_weight * c_loss +
            self.style_weight * s_loss +
            self.identity_weight * id_loss +
            self.tv_weight * tv
        )
        
        # Build loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'content_loss': c_loss.item(),
            'style_loss': s_loss.item(),
            'identity_loss': id_loss.item() if isinstance(id_loss, torch.Tensor) else id_loss,
            'tv_loss': tv.item(),
            'identity_similarity': id_similarity.item() if isinstance(id_similarity, torch.Tensor) else id_similarity,
            # Weighted versions for analysis
            'weighted_content': (self.content_weight * c_loss).item(),
            'weighted_style': (self.style_weight * s_loss).item(),
            'weighted_identity': (self.identity_weight * id_loss).item() if isinstance(id_loss, torch.Tensor) else 0,
            'weighted_tv': (self.tv_weight * tv).item(),
        }
        
        return total_loss, loss_dict
    
    def to(self, device):
        """Move all components to device."""
        super().to(device)
        self.encoder = self.encoder.to(device)
        if self.identity_loss is not None:
            self.identity_loss = self.identity_loss.to(device)
        return self


class LPIPSLoss(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) loss.
    Used for evaluation, not typically for training.
    """
    
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        try:
            import lpips
            self.lpips = lpips.LPIPS(net='alex')
            print("LPIPS model loaded")
        except ImportError:
            print("Warning: lpips not installed")
            self.lpips = None
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS distance."""
        if self.lpips is None:
            return torch.tensor(0.0, device=x.device)
        
        # LPIPS expects images in [-1, 1]
        x = x * 2 - 1
        y = y * 2 - 1
        
        return self.lpips(x, y).mean()


if __name__ == "__main__":
    # Test loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create test tensors
    batch_size = 2
    generated = torch.rand(batch_size, 3, 512, 512).to(device)
    content = torch.rand(batch_size, 3, 512, 512).to(device)
    style = torch.rand(batch_size, 3, 512, 512).to(device)
    adain_features = torch.rand(batch_size, 512, 64, 64).to(device)
    
    # Test individual losses
    gram = GramMatrix()
    test_feat = torch.rand(2, 256, 64, 64).to(device)
    gram_out = gram(test_feat)
    print(f"\nGram Matrix:")
    print(f"  Input: {test_feat.shape}")
    print(f"  Output: {gram_out.shape}")
    
    # Test TV loss
    tv_loss = TotalVariationLoss()
    tv = tv_loss(generated)
    print(f"\nTV Loss: {tv.item():.6f}")
    
    # Test combined loss
    print("\nTesting Combined Loss...")
    combined_loss = CombinedLoss(use_identity_loss=True).to(device)
    
    total_loss, loss_dict = combined_loss(generated, content, style, adain_features)
    
    print("\nLoss Values:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.6f}")
