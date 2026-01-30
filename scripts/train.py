"""Main training script for 3D Shape Generation."""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import random
import numpy as np
from omegaconf import OmegaConf

from src.models.gan_models import create_model, initialize_weights, count_parameters
from src.data.datasets import SyntheticDataset, create_dataloader
from src.train.trainer import Trainer
from src.utils.visualization import TrainingVisualizer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_type: str = "auto") -> str:
    """Get the best available device.
    
    Args:
        device_type: Type of device to use
        
    Returns:
        Device string
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    else:
        return device_type


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config: dict) -> tuple:
    """Create training and validation datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_config = config['data']
    
    if data_config['type'] == 'synthetic':
        # Create synthetic dataset
        train_dataset = SyntheticDataset(
            num_samples=data_config['num_samples'],
            shape_types=data_config['shape_types'],
            num_points=config['model']['num_points'],
            voxel_size=config['model']['voxel_size'],
            data_type='point_cloud' if config['model']['type'] != 'voxel' else 'voxel'
        )
        
        # Split into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
    else:
        raise ValueError(f"Unsupported data type: {data_config['type']}")
    
    return train_dataset, val_dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train 3D Shape Generation GAN')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_dataset(config)
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create models
    print("Creating models...")
    generator, discriminator = create_model(
        model_type=config['model']['type'],
        latent_dim=config['model']['latent_dim'],
        num_points=config['model']['num_points'],
        voxel_size=config['model']['voxel_size'],
        hidden_dims=tuple(config['model']['hidden_dims']),
        use_batchnorm=config['model']['use_batchnorm'],
        dropout=config['model']['dropout']
    )
    
    # Initialize weights
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    # Print model info
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")
    
    # Create trainer
    trainer_config = config['training'].copy()
    trainer_config.update({
        'latent_dim': config['model']['latent_dim'],
        'checkpoint_dir': config['paths']['checkpoint_dir'],
        'log_dir': config['paths']['log_dir']
    })
    
    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
        trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        trainer.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        trainer.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        trainer.history = checkpoint['history']
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Create final visualizations
    print("Creating visualizations...")
    visualizer = TrainingVisualizer(config['paths']['assets_dir'])
    visualizer.plot_training_curves(trainer.history)
    visualizer.create_generation_grid(
        trainer.generator,
        num_samples=16,
        latent_dim=config['model']['latent_dim'],
        device=device
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
