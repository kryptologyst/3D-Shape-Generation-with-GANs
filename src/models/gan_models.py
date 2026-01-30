"""3D Shape Generation Models

This module contains implementations of 3D GAN models for shape generation,
including both point cloud and voxel-based approaches.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointCloudGenerator(nn.Module):
    """Generator for 3D point cloud generation using MLP architecture.
    
    This generator takes a latent vector and produces a 3D point cloud.
    It uses a multi-layer perceptron with batch normalization and leaky ReLU.
    
    Args:
        latent_dim: Dimension of the input latent vector
        point_dim: Dimension of each point (typically 3 for xyz coordinates)
        num_points: Number of points in the generated point cloud
        hidden_dims: List of hidden layer dimensions
        use_batchnorm: Whether to use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        point_dim: int = 3,
        num_points: int = 2048,
        hidden_dims: Tuple[int, ...] = (512, 1024, 2048),
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.point_dim = point_dim
        self.num_points = num_points
        self.output_dim = num_points * point_dim
        
        # Build the generator network
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate point cloud from latent vector.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated point cloud of shape (batch_size, num_points, point_dim)
        """
        batch_size = z.size(0)
        
        # Generate point cloud
        points = self.network(z)
        
        # Reshape to (batch_size, num_points, point_dim)
        points = points.view(batch_size, self.num_points, self.point_dim)
        
        return points


class PointCloudDiscriminator(nn.Module):
    """Discriminator for 3D point cloud discrimination.
    
    This discriminator takes a point cloud and outputs a probability score.
    It uses PointNet-like architecture with max pooling for permutation invariance.
    
    Args:
        point_dim: Dimension of each point (typically 3 for xyz coordinates)
        num_points: Number of points in the input point cloud
        hidden_dims: List of hidden layer dimensions
        use_batchnorm: Whether to use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        point_dim: int = 3,
        num_points: int = 2048,
        hidden_dims: Tuple[int, ...] = (64, 128, 256, 512),
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.point_dim = point_dim
        self.num_points = num_points
        
        # Point-wise feature extraction
        layers = []
        prev_dim = point_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv1d(prev_dim, hidden_dim, 1))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.point_net = nn.Sequential(*layers)
        
        # Global feature aggregation and classification
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Discriminate point cloud.
        
        Args:
            points: Point cloud of shape (batch_size, num_points, point_dim)
            
        Returns:
            Probability score of shape (batch_size, 1)
        """
        batch_size = points.size(0)
        
        # Transpose for Conv1d: (batch_size, point_dim, num_points)
        points = points.transpose(1, 2)
        
        # Extract point-wise features
        features = self.point_net(points)
        
        # Global pooling
        global_features = self.global_pool(features).squeeze(-1)
        
        # Classification
        score = self.classifier(global_features)
        
        return score


class VoxelGenerator(nn.Module):
    """Generator for 3D voxel generation using 3D convolutions.
    
    This generator takes a latent vector and produces a 3D voxel grid.
    It uses transposed 3D convolutions with batch normalization.
    
    Args:
        latent_dim: Dimension of the input latent vector
        voxel_size: Size of the output voxel grid (assumed to be cubic)
        hidden_dims: List of hidden layer dimensions
        use_batchnorm: Whether to use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        voxel_size: int = 64,
        hidden_dims: Tuple[int, ...] = (512, 256, 128, 64),
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.voxel_size = voxel_size
        
        # Calculate the starting size for transposed convolutions
        # We'll start from a small cube and upscale
        start_size = voxel_size // (2 ** (len(hidden_dims) - 1))
        start_channels = hidden_dims[0]
        
        # Initial projection from latent to 3D feature map
        self.initial_proj = nn.Linear(latent_dim, start_channels * start_size ** 3)
        self.start_size = start_size
        self.start_channels = start_channels
        
        # Transposed convolution layers
        layers = []
        prev_channels = start_channels
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            layers.append(nn.ConvTranspose3d(prev_channels, hidden_dim, 4, 2, 1))
            if use_batchnorm:
                layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))
            prev_channels = hidden_dim
        
        # Output layer
        layers.append(nn.ConvTranspose3d(prev_channels, 1, 4, 2, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate voxel grid from latent vector.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated voxel grid of shape (batch_size, 1, voxel_size, voxel_size, voxel_size)
        """
        batch_size = z.size(0)
        
        # Project to initial 3D feature map
        x = self.initial_proj(z)
        x = x.view(batch_size, self.start_channels, self.start_size, self.start_size, self.start_size)
        
        # Generate voxel grid
        voxels = self.network(x)
        
        return voxels


class VoxelDiscriminator(nn.Module):
    """Discriminator for 3D voxel discrimination.
    
    This discriminator takes a voxel grid and outputs a probability score.
    It uses 3D convolutions with batch normalization.
    
    Args:
        voxel_size: Size of the input voxel grid (assumed to be cubic)
        hidden_dims: List of hidden layer dimensions
        use_batchnorm: Whether to use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        voxel_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 128, 256, 512),
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.voxel_size = voxel_size
        
        # Build the discriminator network
        layers = []
        prev_channels = 1
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv3d(prev_channels, hidden_dim, 4, 2, 1))
            if use_batchnorm:
                layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))
            prev_channels = hidden_dim
        
        # Calculate the size after convolutions
        conv_size = voxel_size // (2 ** len(hidden_dims))
        
        # Global pooling and classification
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_channels, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Discriminate voxel grid.
        
        Args:
            voxels: Voxel grid of shape (batch_size, 1, voxel_size, voxel_size, voxel_size)
            
        Returns:
            Probability score of shape (batch_size, 1)
        """
        return self.network(voxels)


class ImprovedPointCloudGenerator(nn.Module):
    """Improved generator with skip connections and better architecture.
    
    This generator uses a more sophisticated architecture with skip connections
    and progressive generation for better quality point clouds.
    
    Args:
        latent_dim: Dimension of the input latent vector
        point_dim: Dimension of each point (typically 3 for xyz coordinates)
        num_points: Number of points in the generated point cloud
        hidden_dims: List of hidden layer dimensions
        use_batchnorm: Whether to use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        point_dim: int = 3,
        num_points: int = 2048,
        hidden_dims: Tuple[int, ...] = (512, 1024, 2048, 4096),
        use_batchnorm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.point_dim = point_dim
        self.num_points = num_points
        self.output_dim = num_points * point_dim
        
        # Build the generator network with skip connections
        self.layers = nn.ModuleList()
        prev_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        
        # Skip connection layers
        self.skip_layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            skip_layer = nn.Linear(latent_dim, hidden_dim)
            self.skip_layers.append(skip_layer)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate point cloud from latent vector with skip connections.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated point cloud of shape (batch_size, num_points, point_dim)
        """
        batch_size = z.size(0)
        x = z
        
        # Forward pass with skip connections
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Add skip connection
            skip = self.skip_layers[i](z)
            x = x + skip
        
        # Generate point cloud
        points = self.output_layer(x)
        
        # Reshape to (batch_size, num_points, point_dim)
        points = points.view(batch_size, self.num_points, self.point_dim)
        
        return points


def create_model(
    model_type: str = "point_cloud",
    latent_dim: int = 128,
    **kwargs
) -> Tuple[nn.Module, nn.Module]:
    """Create generator and discriminator models.
    
    Args:
        model_type: Type of model ("point_cloud" or "voxel")
        latent_dim: Dimension of the latent vector
        **kwargs: Additional arguments for model creation
        
    Returns:
        Tuple of (generator, discriminator) models
    """
    if model_type == "point_cloud":
        generator = PointCloudGenerator(latent_dim=latent_dim, **kwargs)
        discriminator = PointCloudDiscriminator(**kwargs)
    elif model_type == "voxel":
        generator = VoxelGenerator(latent_dim=latent_dim, **kwargs)
        discriminator = VoxelDiscriminator(**kwargs)
    elif model_type == "improved_point_cloud":
        generator = ImprovedPointCloudGenerator(latent_dim=latent_dim, **kwargs)
        discriminator = PointCloudDiscriminator(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return generator, discriminator


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights using Xavier uniform initialization.
    
    Args:
        model: PyTorch model to initialize
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
