"""3D Shape Generation Package."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models.gan_models import (
    PointCloudGenerator,
    PointCloudDiscriminator,
    VoxelGenerator,
    VoxelDiscriminator,
    ImprovedPointCloudGenerator,
    create_model,
    count_parameters,
    initialize_weights
)

from .data.datasets import (
    PointCloudDataset,
    VoxelDataset,
    SyntheticDataset,
    create_dataloader
)

from .train.trainer import (
    GANLoss,
    ChamferDistance,
    EarthMoverDistance,
    Trainer,
    load_checkpoint,
    plot_training_history
)

from .utils.visualization import (
    PointCloudVisualizer,
    VoxelVisualizer,
    TrainingVisualizer,
    save_point_cloud,
    load_point_cloud
)

__all__ = [
    # Models
    "PointCloudGenerator",
    "PointCloudDiscriminator", 
    "VoxelGenerator",
    "VoxelDiscriminator",
    "ImprovedPointCloudGenerator",
    "create_model",
    "count_parameters",
    "initialize_weights",
    
    # Data
    "PointCloudDataset",
    "VoxelDataset",
    "SyntheticDataset",
    "create_dataloader",
    
    # Training
    "GANLoss",
    "ChamferDistance",
    "EarthMoverDistance",
    "Trainer",
    "load_checkpoint",
    "plot_training_history",
    
    # Visualization
    "PointCloudVisualizer",
    "VoxelVisualizer",
    "TrainingVisualizer",
    "save_point_cloud",
    "load_point_cloud"
]
