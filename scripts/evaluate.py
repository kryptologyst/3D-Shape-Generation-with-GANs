"""Evaluation script for 3D Shape Generation."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from src.models.gan_models import create_model, load_checkpoint
from src.data.datasets import SyntheticDataset, create_dataloader
from src.utils.visualization import PointCloudVisualizer, TrainingVisualizer


def evaluate_model(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    num_samples: int = 1000
) -> dict:
    """Evaluate the trained model.
    
    Args:
        generator: Trained generator model
        discriminator: Trained discriminator model
        test_loader: Test data loader
        device: Device to run on
        num_samples: Number of samples to generate for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    generator.eval()
    discriminator.eval()
    
    metrics = {
        'generator_loss': 0.0,
        'discriminator_loss': 0.0,
        'real_accuracy': 0.0,
        'fake_accuracy': 0.0,
        'chamfer_distance': 0.0,
        'earth_mover_distance': 0.0
    }
    
    num_batches = len(test_loader)
    
    with torch.no_grad():
        for batch_idx, real_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Generate fake data
            noise = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_data = generator(noise)
            
            # Compute discriminator predictions
            d_real_pred = discriminator(real_data)
            d_fake_pred = discriminator(fake_data)
            
            # Compute losses
            from src.train.trainer import GANLoss, ChamferDistance, EarthMoverDistance
            
            gan_loss = GANLoss()
            chamfer_loss = ChamferDistance()
            emd_loss = EarthMoverDistance()
            
            d_real_loss = gan_loss(d_real_pred, True)
            d_fake_loss = gan_loss(d_fake_pred, False)
            d_loss = d_real_loss + d_fake_loss
            
            g_loss = gan_loss(d_fake_pred, True)
            
            # Compute metrics
            real_acc = (d_real_pred > 0.5).float().mean()
            fake_acc = (d_fake_pred < 0.5).float().mean()
            
            chamfer_dist = chamfer_loss(fake_data, real_data)
            emd_dist = emd_loss(fake_data, real_data)
            
            # Update metrics
            metrics['generator_loss'] += g_loss.item()
            metrics['discriminator_loss'] += d_loss.item()
            metrics['real_accuracy'] += real_acc.item()
            metrics['fake_accuracy'] += fake_acc.item()
            metrics['chamfer_distance'] += chamfer_dist.item()
            metrics['earth_mover_distance'] += emd_dist.item()
    
    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches
    
    return metrics


def generate_samples(
    generator: torch.nn.Module,
    num_samples: int,
    latent_dim: int,
    device: str,
    save_dir: str
) -> None:
    """Generate and save sample point clouds.
    
    Args:
        generator: Trained generator model
        num_samples: Number of samples to generate
        latent_dim: Dimension of latent vector
        device: Device to run on
        save_dir: Directory to save samples
    """
    generator.eval()
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        generated_samples = generator(noise)
        
        for i, sample in enumerate(generated_samples):
            # Save as numpy array
            np.save(save_path / f"sample_{i:04d}.npy", sample.cpu().numpy())
            
            # Save as PLY file
            from src.utils.visualization import save_point_cloud
            save_point_cloud(sample, str(save_path / f"sample_{i:04d}.ply"))


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate 3D Shape Generation GAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--save-samples', action='store_true',
                       help='Save generated samples')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create models
    generator, discriminator = create_model(
        model_type=config['model']['type'],
        latent_dim=config['model']['latent_dim'],
        num_points=config['model']['num_points'],
        voxel_size=config['model']['voxel_size'],
        hidden_dims=tuple(config['model']['hidden_dims']),
        use_batchnorm=config['model']['use_batchnorm'],
        dropout=config['model']['dropout']
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, generator, discriminator)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Move to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Create test dataset
    test_dataset = SyntheticDataset(
        num_samples=1000,
        shape_types=config['data']['shape_types'],
        num_points=config['model']['num_points'],
        voxel_size=config['model']['voxel_size'],
        data_type='point_cloud' if config['model']['type'] != 'voxel' else 'voxel'
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(generator, discriminator, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate samples if requested
    if args.save_samples:
        print(f"Generating {args.num_samples} samples...")
        generate_samples(
            generator,
            args.num_samples,
            config['model']['latent_dim'],
            device,
            str(output_dir / 'generated_samples')
        )
        print(f"Samples saved to {output_dir / 'generated_samples'}")
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = TrainingVisualizer(str(output_dir))
    visualizer.create_generation_grid(
        generator,
        num_samples=16,
        latent_dim=config['model']['latent_dim'],
        device=device,
        save_path=str(output_dir / 'generation_grid.png')
    )
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
