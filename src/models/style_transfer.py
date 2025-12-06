"""
Complete Style Transfer Network
Combines encoder, AdaIN, and decoder into a single model.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.encoder import VGGEncoder
from src.models.adain import AdaIN, ConditionalAdaIN, StyleMixer
from src.models.decoder import Decoder, ImprovedDecoder
from src.config import NUM_ARTISTS, STYLE_EMBEDDING_DIM


class StyleTransferNetwork(nn.Module):
    """
    Complete neural style transfer network.
    
    Architecture:
    1. VGG19 Encoder (frozen) - extracts features from content and style
    2. AdaIN - transfers style statistics to content features
    3. Decoder (trainable) - reconstructs stylized image
    
    Supports:
    - Basic AdaIN style transfer
    - Conditional (artist-specific) style transfer
    - Style strength control
    """
    
    def __init__(
        self,
        use_conditional: bool = True,
        use_improved_decoder: bool = False,
        num_artists: int = NUM_ARTISTS,
        style_dim: int = STYLE_EMBEDDING_DIM
    ):
        super(StyleTransferNetwork, self).__init__()
        
        self.use_conditional = use_conditional
        
        # Encoder (frozen VGG19)
        self.encoder = VGGEncoder(requires_grad=False)
        
        # AdaIN module
        if use_conditional:
            self.adain = ConditionalAdaIN(
                feature_dim=512,
                style_dim=style_dim,
                num_artists=num_artists
            )
        else:
            self.adain = AdaIN()
        
        # Style mixer for strength control
        self.style_mixer = StyleMixer()
        
        # Decoder (trainable)
        if use_improved_decoder:
            self.decoder = ImprovedDecoder()
        else:
            self.decoder = Decoder(use_residual=True)
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode image to VGG features."""
        return self.encoder(x, return_all=True)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to image."""
        return self.decoder(features)
    
    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        artist_idx: Optional[torch.Tensor] = None,
        style_strength: float = 1.0,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass for style transfer.
        
        Args:
            content: Content image [B, 3, H, W]
            style: Style image [B, 3, H, W]
            artist_idx: Artist indices [B] (required if use_conditional=True)
            style_strength: Style application strength (0.0-1.0)
            return_features: If True, also return intermediate features
        
        Returns:
            stylized: Stylized image [B, 3, H, W]
            features: Dict of intermediate features (if return_features=True)
        """
        # Extract features
        content_features = self.encode(content)
        style_features = self.encode(style)
        
        # Get relu4_1 features for AdaIN
        content_feat = content_features['relu4_1']
        style_feat = style_features['relu4_1']
        
        # Apply AdaIN
        if self.use_conditional:
            if artist_idx is None:
                raise ValueError("artist_idx required when use_conditional=True")
            adain_features = self.adain(content_feat, style_feat, artist_idx)
        else:
            adain_features = self.adain(content_feat, style_feat)
        
        # Apply style strength (blend between content and stylized features)
        if style_strength < 1.0:
            blended_features = self.style_mixer(
                content_feat, style_feat, strength=style_strength
            )
            # Re-apply AdaIN with blended strength
            if self.use_conditional:
                adain_features = self.adain(content_feat, blended_features, artist_idx)
            else:
                adain_features = self.style_mixer(
                    content_feat, adain_features, strength=style_strength
                )
        
        # Decode to image
        stylized = self.decode(adain_features)
        
        if return_features:
            features = {
                'content_features': content_features,
                'style_features': style_features,
                'adain_features': adain_features,
            }
            return stylized, features
        
        return stylized, None
    
    def stylize(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        artist_idx: Optional[torch.Tensor] = None,
        style_strength: float = 1.0
    ) -> torch.Tensor:
        """
        Simplified inference method.
        
        Args:
            content: Content image
            style: Style image
            artist_idx: Artist index
            style_strength: Style strength
        
        Returns:
            Stylized image
        """
        stylized, _ = self.forward(
            content, style, artist_idx, style_strength, return_features=False
        )
        return stylized
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (decoder + conditional adain if used)."""
        params = list(self.decoder.parameters())
        if self.use_conditional:
            params.extend(list(self.adain.parameters()))
        return params
    
    def freeze_encoder(self):
        """Ensure encoder is frozen."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def get_model_size(self) -> Dict[str, int]:
        """Get parameter counts for different components."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        adain_params = sum(p.numel() for p in self.adain.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'encoder': encoder_params,
            'adain': adain_params,
            'decoder': decoder_params,
            'total': encoder_params + adain_params + decoder_params,
            'trainable': trainable
        }


class StyleTransferLoss(nn.Module):
    """
    Combined loss function for style transfer training.
    Includes content, style, and total variation losses.
    """
    
    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 10.0,
        tv_weight: float = 1e-6
    ):
        super(StyleTransferLoss, self).__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        self.mse_loss = nn.MSELoss()
        self.encoder = VGGEncoder(requires_grad=False)
    
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style loss."""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def content_loss(
        self,
        generated_features: Dict[str, torch.Tensor],
        content_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute content loss at relu4_1."""
        return self.mse_loss(
            generated_features['relu4_1'],
            content_features['relu4_1']
        )
    
    def style_loss(
        self,
        generated_features: Dict[str, torch.Tensor],
        style_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute style loss using Gram matrices."""
        loss = 0.0
        for layer in ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']:
            gen_gram = self.gram_matrix(generated_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += self.mse_loss(gen_gram, style_gram)
        return loss
    
    def tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss for smoothness."""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()
    
    def forward(
        self,
        generated: torch.Tensor,
        content: torch.Tensor,
        style: torch.Tensor,
        adain_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.
        
        Args:
            generated: Generated/stylized image
            content: Original content image
            style: Style reference image
            adain_features: Features after AdaIN (target for decoder)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual losses
        """
        # Get features for all images
        gen_features = self.encoder(generated, return_all=True)
        content_features = self.encoder(content, return_all=True)
        style_features = self.encoder(style, return_all=True)
        
        # Content loss: match relu4_1 features to AdaIN output
        # This ensures decoder learns to reconstruct from AdaIN features
        c_loss = self.mse_loss(gen_features['relu4_1'], adain_features)
        
        # Style loss: match Gram matrices at multiple layers
        s_loss = self.style_loss(gen_features, style_features)
        
        # Total variation loss
        tv = self.tv_loss(generated)
        
        # Combine losses
        total_loss = (
            self.content_weight * c_loss +
            self.style_weight * s_loss +
            self.tv_weight * tv
        )
        
        loss_dict = {
            'content_loss': c_loss,
            'style_loss': s_loss,
            'tv_loss': tv,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the complete model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create model
    model = StyleTransferNetwork(use_conditional=True).to(device)
    model.eval()
    
    # Print model size
    sizes = model.get_model_size()
    print("\nModel Parameter Counts:")
    for name, count in sizes.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    batch_size = 2
    content = torch.randn(batch_size, 3, 512, 512).to(device)
    style = torch.randn(batch_size, 3, 512, 512).to(device)
    artist_idx = torch.tensor([0, 1]).to(device)  # Picasso and Rembrandt
    
    with torch.no_grad():
        stylized, features = model(
            content, style, artist_idx,
            style_strength=1.0,
            return_features=True
        )
    
    print(f"\nForward Pass Test:")
    print(f"  Content: {content.shape}")
    print(f"  Style: {style.shape}")
    print(f"  Stylized: {stylized.shape}")
    print(f"  Output range: [{stylized.min():.3f}, {stylized.max():.3f}]")
    
    # Test with different style strengths
    for strength in [0.0, 0.5, 1.0]:
        with torch.no_grad():
            result = model.stylize(content, style, artist_idx, style_strength=strength)
        print(f"  Style strength {strength}: {result.shape}")
    
    # Test loss computation
    loss_fn = StyleTransferLoss().to(device)
    
    # Simulate training step
    model.train()
    stylized, feat = model(content, style, artist_idx, return_features=True)
    
    total_loss, loss_dict = loss_fn(
        stylized, content, style, feat['adain_features']
    )
    
    print(f"\nLoss Computation Test:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.6f}")
