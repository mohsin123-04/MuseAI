"""
Decoder Network for Neural Style Transfer
Reconstructs stylized images from AdaIN-transformed features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


class ReflectionPad2d(nn.Module):
    """Reflection padding to reduce boundary artifacts."""
    
    def __init__(self, padding: int):
        super(ReflectionPad2d, self).__init__()
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, [self.padding] * 4, mode='reflect')


class ConvBlock(nn.Module):
    """
    Convolutional block with reflection padding.
    Uses Instance Normalization instead of Batch Normalization
    for better style transfer results.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        upsample: bool = False,
        activation: bool = True
    ):
        super(ConvBlock, self).__init__()
        
        self.upsample = upsample
        
        layers = []
        
        # Reflection padding
        padding = kernel_size // 2
        layers.append(ReflectionPad2d(padding))
        
        # Convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        
        # Activation (ReLU for hidden layers)
        if activation:
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block for the decoder to help with detail preservation."""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.ReLU(inplace=True),
            ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Decoder(nn.Module):
    """
    Decoder network that mirrors VGG encoder structure.
    
    Takes features from relu4_1 (512 channels, 64x64 for 512px input)
    and reconstructs the stylized image.
    
    Architecture mirrors VGG but in reverse:
    - Uses nearest neighbor upsampling instead of pooling
    - Uses reflection padding to reduce artifacts
    - No batch normalization (can interfere with style statistics)
    """
    
    def __init__(self, use_residual: bool = True):
        super(Decoder, self).__init__()
        
        self.use_residual = use_residual
        
        # Optional residual blocks for better reconstruction
        if use_residual:
            self.res_blocks = nn.Sequential(
                ResidualBlock(512),
                ResidualBlock(512),
            )
        
        # Decoder layers (mirror of VGG encoder from relu4_1)
        # relu4_1: 512 channels, 64x64 -> relu3_1: 256 channels, 128x128
        self.decoder = nn.Sequential(
            # Block 4 (reverse)
            ConvBlock(512, 256, kernel_size=3),
            ConvBlock(256, 256, kernel_size=3, upsample=True),
            
            # Block 3 (reverse)
            ConvBlock(256, 256, kernel_size=3),
            ConvBlock(256, 256, kernel_size=3),
            ConvBlock(256, 128, kernel_size=3, upsample=True),
            
            # Block 2 (reverse)
            ConvBlock(128, 128, kernel_size=3),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            
            # Block 1 (reverse)
            ConvBlock(64, 64, kernel_size=3),
            
            # Final output layer (no activation, use sigmoid later)
            ConvBlock(64, 3, kernel_size=3, activation=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode features to image.
        
        Args:
            x: Features from AdaIN [B, 512, H, W]
        
        Returns:
            Reconstructed image [B, 3, H*8, W*8] in range [0, 1]
        """
        if self.use_residual:
            x = self.res_blocks(x)
        
        x = self.decoder(x)
        
        # Clamp output to valid range
        x = torch.sigmoid(x)
        
        return x


class ImprovedDecoder(nn.Module):
    """
    Improved decoder with better architecture for quality outputs.
    Uses more sophisticated upsampling and skip-connection-like refinement.
    """
    
    def __init__(self):
        super(ImprovedDecoder, self).__init__()
        
        # Initial processing
        self.initial = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        
        # Final output
        self.final = nn.Sequential(
            ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.initial(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    # Test decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test basic decoder
    decoder = Decoder().to(device)
    features = torch.randn(2, 512, 64, 64).to(device)  # Features from relu4_1
    
    output = decoder(features)
    print(f"Decoder Test:")
    print(f"  Input features: {features.shape}")
    print(f"  Output image: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test improved decoder
    improved_decoder = ImprovedDecoder().to(device)
    output_improved = improved_decoder(features)
    print(f"\nImproved Decoder Test:")
    print(f"  Output image: {output_improved.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in decoder.parameters())
    params_improved = sum(p.numel() for p in improved_decoder.parameters())
    print(f"\nParameter count:")
    print(f"  Basic decoder: {params:,}")
    print(f"  Improved decoder: {params_improved:,}")
