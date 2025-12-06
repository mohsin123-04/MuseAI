"""
Adaptive Instance Normalization (AdaIN) for Style Transfer
Core mechanism for blending content and style features.
"""
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import NUM_ARTISTS, STYLE_EMBEDDING_DIM


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer.
    
    Transfers the mean and variance of style features to content features,
    which is the core mechanism of neural style transfer.
    
    Formula: AdaIN(content, style) = σ(style) * ((content - μ(content)) / σ(content)) + μ(style)
    """
    
    def __init__(self, eps: float = 1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps
    
    def calc_mean_std(self, x: torch.Tensor) -> tuple:
        """
        Calculate mean and standard deviation per instance and channel.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            mean: [B, C, 1, 1]
            std: [B, C, 1, 1]
        """
        b, c, h, w = x.size()
        
        # Calculate mean and variance across spatial dimensions
        mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        std = x.view(b, c, -1).std(dim=2).view(b, c, 1, 1) + self.eps
        
        return mean, std
    
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN transformation.
        
        Args:
            content: Content features [B, C, H, W]
            style: Style features [B, C, H, W]
        
        Returns:
            Normalized features with style statistics [B, C, H, W]
        """
        content_mean, content_std = self.calc_mean_std(content)
        style_mean, style_std = self.calc_mean_std(style)
        
        # Normalize content features
        normalized = (content - content_mean) / content_std
        
        # Apply style statistics
        output = normalized * style_std + style_mean
        
        return output


class ConditionalAdaIN(nn.Module):
    """
    Conditional Adaptive Instance Normalization.
    
    Extends AdaIN to support artist-specific style embeddings,
    allowing the model to learn different style transformations
    for Picasso vs Rembrandt.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        style_dim: int = STYLE_EMBEDDING_DIM,
        num_artists: int = NUM_ARTISTS
    ):
        super(ConditionalAdaIN, self).__init__()
        
        self.feature_dim = feature_dim
        self.style_dim = style_dim
        self.num_artists = num_artists
        self.eps = 1e-5
        
        # Artist-specific style embeddings
        self.artist_embeddings = nn.Embedding(num_artists, style_dim)
        
        # MLPs to predict affine parameters from style features + artist embedding
        # These learn to modulate the style statistics based on the artist
        self.gamma_net = nn.Sequential(
            nn.Linear(feature_dim + style_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(feature_dim + style_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in [self.gamma_net, self.beta_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # Initialize artist embeddings
        nn.init.normal_(self.artist_embeddings.weight, mean=0, std=0.02)
    
    def calc_mean_std(self, x: torch.Tensor) -> tuple:
        """Calculate mean and standard deviation."""
        b, c, h, w = x.size()
        mean = x.view(b, c, -1).mean(dim=2)
        std = x.view(b, c, -1).std(dim=2) + self.eps
        return mean, std
    
    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        artist_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Conditional AdaIN transformation.
        
        Args:
            content: Content features [B, C, H, W]
            style: Style features [B, C, H, W]
            artist_idx: Artist indices [B] (0=Picasso, 1=Rembrandt)
        
        Returns:
            Stylized features [B, C, H, W]
        """
        b, c, h, w = content.size()
        
        # Get content statistics
        content_mean, content_std = self.calc_mean_std(content)
        
        # Get style statistics
        style_mean, style_std = self.calc_mean_std(style)
        
        # Get artist embedding
        artist_emb = self.artist_embeddings(artist_idx)  # [B, style_dim]
        
        # Combine style statistics with artist embedding
        style_info = torch.cat([style_mean, artist_emb], dim=1)  # [B, feature_dim + style_dim]
        
        # Predict affine parameters
        gamma = self.gamma_net(style_info)  # [B, feature_dim]
        beta = self.beta_net(style_info)    # [B, feature_dim]
        
        # Normalize content
        normalized = (content - content_mean.view(b, c, 1, 1)) / content_std.view(b, c, 1, 1)
        
        # Apply conditional modulation
        # Blend learned parameters with original style statistics
        final_std = style_std.view(b, c, 1, 1) * (1 + gamma.view(b, c, 1, 1) * 0.1)
        final_mean = style_mean.view(b, c, 1, 1) + beta.view(b, c, 1, 1) * 0.1
        
        output = normalized * final_std + final_mean
        
        return output


class StyleMixer(nn.Module):
    """
    Mix content and style with adjustable strength.
    Allows users to control how much style is applied.
    """
    
    def __init__(self, eps: float = 1e-5):
        super(StyleMixer, self).__init__()
        self.adain = AdaIN(eps=eps)
    
    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Apply style with adjustable strength.
        
        Args:
            content: Content features
            style: Style features
            strength: Style strength (0.0 = content only, 1.0 = full style)
        
        Returns:
            Blended features
        """
        stylized = self.adain(content, style)
        
        # Interpolate between content and stylized
        output = (1 - strength) * content + strength * stylized
        
        return output


if __name__ == "__main__":
    # Test AdaIN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Basic AdaIN test
    adain = AdaIN().to(device)
    content = torch.randn(2, 512, 64, 64).to(device)
    style = torch.randn(2, 512, 64, 64).to(device)
    
    output = adain(content, style)
    print(f"AdaIN Test:")
    print(f"  Content: {content.shape}")
    print(f"  Style: {style.shape}")
    print(f"  Output: {output.shape}")
    
    # Conditional AdaIN test
    cond_adain = ConditionalAdaIN(feature_dim=512).to(device)
    artist_idx = torch.tensor([0, 1]).to(device)  # Picasso and Rembrandt
    
    output = cond_adain(content, style, artist_idx)
    print(f"\nConditional AdaIN Test:")
    print(f"  Output: {output.shape}")
    
    # Style mixer test
    mixer = StyleMixer().to(device)
    output_half = mixer(content, style, strength=0.5)
    print(f"\nStyle Mixer Test (strength=0.5):")
    print(f"  Output: {output_half.shape}")
