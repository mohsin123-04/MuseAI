"""
VGG19 Encoder for Neural Style Transfer
Extracts features from images using pretrained VGG19.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights
from collections import OrderedDict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import IMAGENET_MEAN, IMAGENET_STD


class VGGEncoder(nn.Module):
    """
    VGG19 encoder that extracts multi-level features for style transfer.
    
    The encoder is frozen (no gradient updates) and used to:
    1. Extract content features (relu4_1)
    2. Extract style features (relu1_1, relu2_1, relu3_1, relu4_1)
    """
    
    def __init__(self, requires_grad: bool = False):
        super(VGGEncoder, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # We only need the feature extraction layers (not classifier)
        vgg_features = vgg.features
        
        # Define slices to extract features at different layers
        # VGG19 architecture: 
        # 0-1: conv1_1, relu1_1
        # 2-4: conv1_2, relu1_2, maxpool
        # 5-6: conv2_1, relu2_1
        # 7-9: conv2_2, relu2_2, maxpool
        # 10-11: conv3_1, relu3_1
        # ...and so on
        
        self.slice1 = nn.Sequential()  # -> relu1_1
        self.slice2 = nn.Sequential()  # -> relu2_1
        self.slice3 = nn.Sequential()  # -> relu3_1
        self.slice4 = nn.Sequential()  # -> relu4_1
        
        # relu1_1 is at index 1
        for x in range(2):
            self.slice1.add_module(str(x), vgg_features[x])
        
        # relu2_1 is at index 6 (after maxpool)
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_features[x])
        
        # relu3_1 is at index 11
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_features[x])
        
        # relu4_1 is at index 20 (this is where AdaIN operates)
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_features[x])
        
        # Freeze encoder weights
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # Register normalization buffers
        self.register_buffer(
            'mean', 
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', 
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input images with ImageNet stats."""
        return (x - self.mean) / self.std
    
    def forward(self, x: torch.Tensor, return_all: bool = True):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input image tensor [B, 3, H, W] in range [0, 1]
            return_all: If True, return features from all layers.
                       If False, only return relu4_1 features.
        
        Returns:
            If return_all: Dict of features {'relu1_1': ..., 'relu2_1': ..., etc}
            If not return_all: Features from relu4_1 only
        """
        # Normalize input
        x = self.normalize(x)
        
        # Extract features at each level
        relu1_1 = self.slice1(x)
        relu2_1 = self.slice2(relu1_1)
        relu3_1 = self.slice3(relu2_1)
        relu4_1 = self.slice4(relu3_1)
        
        if return_all:
            return {
                'relu1_1': relu1_1,
                'relu2_1': relu2_1,
                'relu3_1': relu3_1,
                'relu4_1': relu4_1,
            }
        else:
            return relu4_1
    
    def get_content_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract content features (relu4_1) only."""
        return self.forward(x, return_all=False)
    
    def get_style_features(self, x: torch.Tensor) -> dict:
        """Extract all style features."""
        return self.forward(x, return_all=True)


class VGGLoss(nn.Module):
    """
    Compute perceptual loss using VGG features.
    Used for content and style losses during training.
    """
    
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.encoder = VGGEncoder(requires_grad=False)
        self.mse_loss = nn.MSELoss()
    
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix for style representation.
        
        Args:
            x: Feature tensor [B, C, H, W]
        
        Returns:
            Gram matrix [B, C, C]
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def content_loss(self, generated: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """
        Compute content loss at relu4_1.
        
        Args:
            generated: Generated image
            content: Original content image
        
        Returns:
            Content loss value
        """
        gen_features = self.encoder.get_content_features(generated)
        content_features = self.encoder.get_content_features(content)
        return self.mse_loss(gen_features, content_features)
    
    def style_loss(self, generated: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Compute style loss using Gram matrices at multiple layers.
        
        Args:
            generated: Generated image
            style: Style reference image
        
        Returns:
            Style loss value
        """
        gen_features = self.encoder.get_style_features(generated)
        style_features = self.encoder.get_style_features(style)
        
        loss = 0.0
        for layer in gen_features.keys():
            gen_gram = self.gram_matrix(gen_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += self.mse_loss(gen_gram, style_gram)
        
        return loss


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VGGEncoder().to(device)
    
    # Test with random input
    x = torch.randn(2, 3, 512, 512).to(device)
    features = encoder(x, return_all=True)
    
    print("VGG Encoder Test:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    # Test content features only
    content_feat = encoder.get_content_features(x)
    print(f"\nContent features (relu4_1): {content_feat.shape}")
