#!/usr/bin/env python3
"""Simple example script demonstrating 3D shape generation."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.models.gan_models import create_model, initialize_weights
from src.data.datasets import SyntheticDataset
from src.utils.visualization import PointCloudVisualizer


def main():
    """Main function demonstrating basic usage."""
    print("3D Shape Generation - Simple Example")
    print("=" * 40)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create models
    print("\nCreating models...")
    generator, discriminator = create_model(
        model_type="point_cloud",
        latent_dim=128,
        num_points=1024
    )
    
    # Initialize weights
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    # Move to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create dataset
    print("\nCreating synthetic dataset...")
    dataset = SyntheticDataset(
        num_samples=100,
        shape_types=['sphere', 'cube', 'cylinder'],
        num_points=1024,
        data_type='point_cloud'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize real data
    print("\nVisualizing real data...")
    real_sample = dataset[0]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    points = real_sample.numpy()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=points[:, 2], cmap='viridis', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Real Point Cloud Sample')
    
    plt.tight_layout()
    plt.savefig('real_sample.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate fake data
    print("\nGenerating fake data...")
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(4, 128, device=device)
        fake_samples = generator(noise)
    
    # Visualize generated data
    print("Visualizing generated data...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for i, sample in enumerate(fake_samples):
        points = sample.cpu().numpy()
        
        axes[i].scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=points[:, 2], cmap='viridis', s=1)
        axes[i].set_title(f'Generated Sample {i+1}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_zlabel('Z')
    
    plt.suptitle('Generated 3D Shapes')
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test discriminator
    print("\nTesting discriminator...")
    discriminator.eval()
    with torch.no_grad():
        # Test on real data
        real_batch = torch.stack([dataset[i] for i in range(4)]).to(device)
        real_pred = discriminator(real_batch)
        
        # Test on fake data
        fake_pred = discriminator(fake_samples)
        
        print(f"Real data predictions: {real_pred.cpu().numpy().flatten()}")
        print(f"Fake data predictions: {fake_pred.cpu().numpy().flatten()}")
    
    print("\nExample completed successfully!")
    print("Check 'real_sample.png' and 'generated_samples.png' for visualizations.")


if __name__ == "__main__":
    main()
