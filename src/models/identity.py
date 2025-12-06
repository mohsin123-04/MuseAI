"""
FaceNet Identity Preservation Module
Uses pretrained FaceNet to ensure facial identity is preserved in stylized images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import FACENET_MODEL, FACENET_INPUT_SIZE, IDENTITY_EMBEDDING_DIM


class FaceNetIdentity(nn.Module):
    """
    FaceNet-based identity loss module.
    
    Uses InceptionResnetV1 pretrained on VGGFace2 to extract face embeddings.
    The identity loss ensures that the stylized face maintains the same
    identity as the original content face.
    """
    
    def __init__(
        self,
        pretrained: str = FACENET_MODEL,
        device: Optional[torch.device] = None
    ):
        super(FaceNetIdentity, self).__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = FACENET_INPUT_SIZE
        
        # Load pretrained FaceNet
        try:
            from facenet_pytorch import InceptionResnetV1
            self.facenet = InceptionResnetV1(pretrained=pretrained).eval()
            print(f"Loaded FaceNet pretrained on {pretrained}")
        except ImportError:
            print("Warning: facenet_pytorch not installed. Using mock embeddings.")
            self.facenet = None
        
        # Freeze FaceNet weights
        if self.facenet is not None:
            for param in self.facenet.parameters():
                param.requires_grad = False
        
        # FaceNet expects specific normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        )
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for FaceNet.
        
        Args:
            x: Images [B, 3, H, W] in range [0, 1]
        
        Returns:
            Preprocessed images [B, 3, 160, 160] normalized to [-1, 1]
        """
        # Resize to 160x160
        x = F.interpolate(
            x, size=(self.input_size, self.input_size),
            mode='bilinear', align_corners=False
        )
        
        # Normalize to [-1, 1]
        x = (x - self.mean) / self.std
        
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embeddings.
        
        Args:
            x: Face images [B, 3, H, W] in range [0, 1]
        
        Returns:
            Embeddings [B, 512]
        """
        if self.facenet is None:
            # Mock embeddings if FaceNet not available
            return torch.randn(x.size(0), IDENTITY_EMBEDDING_DIM, device=x.device)
        
        x = self.preprocess(x)
        
        with torch.no_grad():
            embeddings = self.facenet(x)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(
        self,
        original: torch.Tensor,
        stylized: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute identity preservation loss.
        
        Args:
            original: Original content face [B, 3, H, W]
            stylized: Stylized face [B, 3, H, W]
        
        Returns:
            identity_loss: Mean squared error between embeddings
            similarity: Cosine similarity (for monitoring)
        """
        # Get embeddings
        orig_emb = self.get_embedding(original)
        style_emb = self.get_embedding(stylized)
        
        # MSE loss between embeddings
        identity_loss = F.mse_loss(style_emb, orig_emb)
        
        # Cosine similarity for monitoring
        similarity = F.cosine_similarity(orig_emb, style_emb, dim=1).mean()
        
        return identity_loss, similarity
    
    def compute_similarity(
        self,
        face1: torch.Tensor,
        face2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two faces.
        
        Args:
            face1: First face image
            face2: Second face image
        
        Returns:
            Cosine similarity score
        """
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)
        
        return F.cosine_similarity(emb1, emb2, dim=1)


class IdentityLoss(nn.Module):
    """
    Wrapper for identity loss computation during training.
    """
    
    def __init__(self, weight: float = 1.0):
        super(IdentityLoss, self).__init__()
        self.weight = weight
        self.facenet = FaceNetIdentity()
    
    def forward(
        self,
        original: torch.Tensor,
        stylized: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute weighted identity loss.
        
        Returns:
            loss: Weighted identity loss
            metrics: Dictionary with loss and similarity
        """
        identity_loss, similarity = self.facenet(original, stylized)
        
        weighted_loss = self.weight * identity_loss
        
        metrics = {
            'identity_loss': identity_loss.item(),
            'identity_similarity': similarity.item(),
            'weighted_identity_loss': weighted_loss.item()
        }
        
        return weighted_loss, metrics
    
    def to(self, device):
        """Move to device."""
        super().to(device)
        self.facenet = self.facenet.to(device)
        return self


if __name__ == "__main__":
    # Test identity module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create module
    identity = FaceNetIdentity().to(device)
    
    # Test with random images (simulating faces)
    original = torch.rand(2, 3, 512, 512).to(device)
    stylized = torch.rand(2, 3, 512, 512).to(device)
    
    # Get embeddings
    orig_emb = identity.get_embedding(original)
    print(f"\nEmbedding shape: {orig_emb.shape}")
    print(f"Embedding norm: {orig_emb.norm(dim=1)}")  # Should be ~1 after L2 norm
    
    # Compute loss
    loss, similarity = identity(original, stylized)
    print(f"\nIdentity Loss: {loss.item():.6f}")
    print(f"Similarity: {similarity.item():.6f}")
    
    # Test with slightly perturbed version (should have high similarity)
    perturbed = original + 0.1 * torch.randn_like(original)
    perturbed = perturbed.clamp(0, 1)
    
    loss_perturbed, sim_perturbed = identity(original, perturbed)
    print(f"\nPerturbed version:")
    print(f"  Identity Loss: {loss_perturbed.item():.6f}")
    print(f"  Similarity: {sim_perturbed.item():.6f}")
    
    # Test IdentityLoss wrapper
    id_loss = IdentityLoss(weight=5.0).to(device)
    weighted_loss, metrics = id_loss(original, stylized)
    print(f"\nWeighted Identity Loss:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
