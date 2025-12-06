"""
Evaluation Metrics for MuseAI
Implements SSIM, LPIPS, Gram distance, and identity similarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


class SSIMMetric:
    """
    Structural Similarity Index (SSIM) for content preservation.
    Higher SSIM = more structural similarity to original content.
    """
    
    def __init__(self, window_size: int = 11, channel: int = 3):
        self.window_size = window_size
        self.channel = channel
        self.ssim_fn = None
        
        try:
            from pytorch_msssim import ssim, SSIM
            self.ssim_fn = SSIM(
                data_range=1.0,
                size_average=True,
                channel=channel,
                win_size=window_size
            )
            print("Using pytorch_msssim for SSIM")
        except ImportError:
            print("Warning: pytorch_msssim not installed, using custom SSIM")
    
    def _gaussian_window(self, size: int, sigma: float = 1.5) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g)
    
    def _custom_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2
    ) -> torch.Tensor:
        """Custom SSIM implementation."""
        # Create window
        window = self._gaussian_window(window_size)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(self.channel, 1, window_size, window_size)
        window = window.to(img1.device, img1.dtype)
        
        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=self.channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=self.channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def __call__(
        self,
        generated: torch.Tensor,
        original: torch.Tensor
    ) -> float:
        """
        Compute SSIM between generated and original images.
        
        Args:
            generated: Generated image [B, 3, H, W]
            original: Original content image [B, 3, H, W]
        
        Returns:
            SSIM score (0-1, higher is better)
        """
        if self.ssim_fn is not None:
            self.ssim_fn = self.ssim_fn.to(generated.device)
            return self.ssim_fn(generated, original).item()
        else:
            return self._custom_ssim(generated, original).item()


class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    Lower LPIPS = more perceptually similar.
    """
    
    def __init__(self):
        self.lpips = None
        
        try:
            import lpips
            self.lpips = lpips.LPIPS(net='alex', verbose=False)
            print("LPIPS metric initialized")
        except ImportError:
            print("Warning: lpips not installed")
    
    def __call__(
        self,
        generated: torch.Tensor,
        original: torch.Tensor
    ) -> float:
        """
        Compute LPIPS distance.
        
        Args:
            generated: Generated image [B, 3, H, W] in [0, 1]
            original: Original image [B, 3, H, W] in [0, 1]
        
        Returns:
            LPIPS distance (lower is better, typically 0-1)
        """
        if self.lpips is None:
            return 0.0
        
        self.lpips = self.lpips.to(generated.device)
        
        # LPIPS expects images in [-1, 1]
        gen_scaled = generated * 2 - 1
        orig_scaled = original * 2 - 1
        
        with torch.no_grad():
            distance = self.lpips(gen_scaled, orig_scaled)
        
        return distance.mean().item()


class GramDistance:
    """
    Gram matrix distance for style matching.
    Lower distance = better style match.
    """
    
    def __init__(self):
        from src.models.encoder import VGGEncoder
        self.encoder = None
        self._encoder_class = VGGEncoder
    
    def _ensure_encoder(self, device):
        """Lazily initialize encoder on correct device."""
        if self.encoder is None:
            self.encoder = self._encoder_class(requires_grad=False)
        self.encoder = self.encoder.to(device)
    
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix."""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def __call__(
        self,
        generated: torch.Tensor,
        style: torch.Tensor,
        layers: list = None
    ) -> float:
        """
        Compute Gram matrix distance between generated and style.
        
        Args:
            generated: Generated/stylized image
            style: Style reference image
            layers: VGG layers to use
        
        Returns:
            Average Gram distance across layers
        """
        self._ensure_encoder(generated.device)
        
        if layers is None:
            layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        
        with torch.no_grad():
            gen_features = self.encoder(generated, return_all=True)
            style_features = self.encoder(style, return_all=True)
        
        total_distance = 0.0
        for layer in layers:
            gen_gram = self.gram_matrix(gen_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            total_distance += F.mse_loss(gen_gram, style_gram).item()
        
        return total_distance / len(layers)


class IdentitySimilarity:
    """
    FaceNet embedding similarity for identity preservation.
    Higher similarity = better identity preservation.
    """
    
    def __init__(self):
        from src.models.identity import FaceNetIdentity
        self.facenet = None
        self._facenet_class = FaceNetIdentity
    
    def _ensure_facenet(self, device):
        """Lazily initialize FaceNet on correct device."""
        if self.facenet is None:
            self.facenet = self._facenet_class()
        self.facenet = self.facenet.to(device)
    
    def __call__(
        self,
        generated: torch.Tensor,
        original: torch.Tensor
    ) -> float:
        """
        Compute identity similarity.
        
        Args:
            generated: Generated/stylized face
            original: Original content face
        
        Returns:
            Cosine similarity (-1 to 1, higher is better)
        """
        self._ensure_facenet(generated.device)
        
        with torch.no_grad():
            similarity = self.facenet.compute_similarity(original, generated)
        
        return similarity.mean().item()


def compute_metrics(
    generated: torch.Tensor,
    content: torch.Tensor,
    style: torch.Tensor,
    metrics: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        generated: Generated/stylized image
        content: Original content image
        style: Style reference image
        metrics: Optional dict of pre-initialized metric objects
    
    Returns:
        Dictionary of metric values
    """
    if metrics is None:
        metrics = {
            'ssim': SSIMMetric(),
            'lpips': LPIPSMetric(),
            'gram': GramDistance(),
            'identity': IdentitySimilarity()
        }
    
    results = {}
    
    # SSIM (content preservation)
    try:
        results['ssim'] = metrics['ssim'](generated, content)
    except Exception as e:
        print(f"SSIM error: {e}")
        results['ssim'] = 0.0
    
    # LPIPS (perceptual similarity to content)
    try:
        results['lpips'] = metrics['lpips'](generated, content)
    except Exception as e:
        print(f"LPIPS error: {e}")
        results['lpips'] = 0.0
    
    # Gram distance (style matching)
    try:
        results['gram_distance'] = metrics['gram'](generated, style)
    except Exception as e:
        print(f"Gram error: {e}")
        results['gram_distance'] = 0.0
    
    # Identity similarity
    try:
        results['identity_similarity'] = metrics['identity'](generated, content)
    except Exception as e:
        print(f"Identity error: {e}")
        results['identity_similarity'] = 0.0
    
    return results


if __name__ == "__main__":
    # Test metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing metrics on {device}")
    
    # Create test images
    batch_size = 2
    generated = torch.rand(batch_size, 3, 256, 256).to(device)
    content = torch.rand(batch_size, 3, 256, 256).to(device)
    style = torch.rand(batch_size, 3, 256, 256).to(device)
    
    # Test individual metrics
    print("\nTesting individual metrics:")
    
    ssim_metric = SSIMMetric()
    ssim_val = ssim_metric(generated, content)
    print(f"  SSIM: {ssim_val:.4f}")
    
    lpips_metric = LPIPSMetric()
    lpips_val = lpips_metric(generated, content)
    print(f"  LPIPS: {lpips_val:.4f}")
    
    gram_metric = GramDistance()
    gram_val = gram_metric(generated, style)
    print(f"  Gram Distance: {gram_val:.4f}")
    
    identity_metric = IdentitySimilarity()
    identity_val = identity_metric(generated, content)
    print(f"  Identity Similarity: {identity_val:.4f}")
    
    # Test combined
    print("\nTesting combined metrics:")
    all_metrics = compute_metrics(generated, content, style)
    for name, value in all_metrics.items():
        print(f"  {name}: {value:.4f}")
