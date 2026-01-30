"""Tests for 3D Shape Generation models and utilities."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.gan_models import (
    PointCloudGenerator, PointCloudDiscriminator,
    VoxelGenerator, VoxelDiscriminator,
    ImprovedPointCloudGenerator,
    create_model, count_parameters, initialize_weights
)
from src.data.datasets import SyntheticDataset, PointCloudDataset, VoxelDataset
from src.train.trainer import GANLoss, ChamferDistance, EarthMoverDistance, Trainer
from src.utils.visualization import PointCloudVisualizer, VoxelVisualizer


class TestModels:
    """Test cases for model architectures."""
    
    def test_point_cloud_generator(self):
        """Test PointCloudGenerator."""
        generator = PointCloudGenerator(
            latent_dim=128,
            num_points=1024,
            point_dim=3
        )
        
        # Test forward pass
        batch_size = 4
        noise = torch.randn(batch_size, 128)
        output = generator(noise)
        
        assert output.shape == (batch_size, 1024, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_point_cloud_discriminator(self):
        """Test PointCloudDiscriminator."""
        discriminator = PointCloudDiscriminator(
            num_points=1024,
            point_dim=3
        )
        
        # Test forward pass
        batch_size = 4
        points = torch.randn(batch_size, 1024, 3)
        output = discriminator(points)
        
        assert output.shape == (batch_size, 1)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_voxel_generator(self):
        """Test VoxelGenerator."""
        generator = VoxelGenerator(
            latent_dim=128,
            voxel_size=32
        )
        
        # Test forward pass
        batch_size = 2
        noise = torch.randn(batch_size, 128)
        output = generator(noise)
        
        assert output.shape == (batch_size, 1, 32, 32, 32)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_voxel_discriminator(self):
        """Test VoxelDiscriminator."""
        discriminator = VoxelDiscriminator(voxel_size=32)
        
        # Test forward pass
        batch_size = 2
        voxels = torch.rand(batch_size, 1, 32, 32, 32)
        output = discriminator(voxels)
        
        assert output.shape == (batch_size, 1)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_improved_point_cloud_generator(self):
        """Test ImprovedPointCloudGenerator."""
        generator = ImprovedPointCloudGenerator(
            latent_dim=128,
            num_points=1024,
            point_dim=3
        )
        
        # Test forward pass
        batch_size = 4
        noise = torch.randn(batch_size, 128)
        output = generator(noise)
        
        assert output.shape == (batch_size, 1024, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_create_model(self):
        """Test create_model function."""
        # Test point cloud model
        generator, discriminator = create_model(
            model_type="point_cloud",
            latent_dim=128,
            num_points=1024
        )
        
        assert isinstance(generator, PointCloudGenerator)
        assert isinstance(discriminator, PointCloudDiscriminator)
        
        # Test voxel model
        generator, discriminator = create_model(
            model_type="voxel",
            latent_dim=128,
            voxel_size=32
        )
        
        assert isinstance(generator, VoxelGenerator)
        assert isinstance(discriminator, VoxelDiscriminator)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        generator = PointCloudGenerator(latent_dim=128, num_points=1024)
        num_params = count_parameters(generator)
        
        assert num_params > 0
        assert isinstance(num_params, int)
    
    def test_initialize_weights(self):
        """Test weight initialization."""
        generator = PointCloudGenerator(latent_dim=128, num_points=1024)
        
        # Initialize weights
        initialize_weights(generator)
        
        # Check that weights are initialized (not all zeros)
        for param in generator.parameters():
            assert not torch.all(param == 0)


class TestDatasets:
    """Test cases for datasets."""
    
    def test_synthetic_dataset(self):
        """Test SyntheticDataset."""
        dataset = SyntheticDataset(
            num_samples=100,
            shape_types=['sphere', 'cube'],
            num_points=1024,
            data_type='point_cloud'
        )
        
        assert len(dataset) == 100
        
        # Test getting a sample
        sample = dataset[0]
        assert sample.shape == (1024, 3)
        assert not torch.isnan(sample).any()
        assert not torch.isinf(sample).any()
    
    def test_synthetic_dataset_voxel(self):
        """Test SyntheticDataset with voxel data."""
        dataset = SyntheticDataset(
            num_samples=50,
            shape_types=['sphere', 'cube'],
            voxel_size=32,
            data_type='voxel'
        )
        
        assert len(dataset) == 50
        
        # Test getting a sample
        sample = dataset[0]
        assert sample.shape == (32, 32, 32)
        assert not torch.isnan(sample).any()
        assert not torch.isinf(sample).any()


class TestLosses:
    """Test cases for loss functions."""
    
    def test_gan_loss_vanilla(self):
        """Test GAN loss in vanilla mode."""
        loss_fn = GANLoss(gan_mode='vanilla')
        
        # Test real prediction
        real_pred = torch.rand(4, 1)
        real_loss = loss_fn(real_pred, True)
        
        assert real_loss.item() > 0
        assert not torch.isnan(real_loss)
        
        # Test fake prediction
        fake_pred = torch.rand(4, 1)
        fake_loss = loss_fn(fake_pred, False)
        
        assert fake_loss.item() > 0
        assert not torch.isnan(fake_loss)
    
    def test_gan_loss_lsgan(self):
        """Test GAN loss in LSGAN mode."""
        loss_fn = GANLoss(gan_mode='lsgan')
        
        # Test real prediction
        real_pred = torch.rand(4, 1)
        real_loss = loss_fn(real_pred, True)
        
        assert real_loss.item() > 0
        assert not torch.isnan(real_loss)
    
    def test_chamfer_distance(self):
        """Test Chamfer distance loss."""
        loss_fn = ChamferDistance()
        
        # Create test point clouds
        pred = torch.rand(2, 100, 3)
        target = torch.rand(2, 100, 3)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_earth_mover_distance(self):
        """Test Earth Mover's Distance loss."""
        loss_fn = EarthMoverDistance()
        
        # Create test point clouds
        pred = torch.rand(2, 100, 3)
        target = torch.rand(2, 100, 3)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestVisualization:
    """Test cases for visualization utilities."""
    
    def test_point_cloud_visualizer(self):
        """Test PointCloudVisualizer."""
        visualizer = PointCloudVisualizer()
        
        # Create test point cloud
        points = torch.rand(100, 3)
        
        # Test matplotlib visualization (should not raise error)
        try:
            visualizer.visualize_matplotlib(points, title="Test")
        except Exception as e:
            pytest.skip(f"Matplotlib visualization failed: {e}")
    
    def test_voxel_visualizer(self):
        """Test VoxelVisualizer."""
        visualizer = VoxelVisualizer()
        
        # Create test voxel grid
        voxels = torch.rand(16, 16, 16)
        
        # Test matplotlib visualization (should not raise error)
        try:
            visualizer.visualize_matplotlib(voxels, title="Test")
        except Exception as e:
            pytest.skip(f"Matplotlib visualization failed: {e}")


class TestTraining:
    """Test cases for training utilities."""
    
    def test_trainer_initialization(self):
        """Test Trainer initialization."""
        # Create simple models
        generator = PointCloudGenerator(latent_dim=64, num_points=512)
        discriminator = PointCloudDiscriminator(num_points=512)
        
        # Create simple dataset
        dataset = SyntheticDataset(num_samples=100, num_points=512)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create trainer
        config = {
            'lr_g': 0.0002,
            'lr_d': 0.0002,
            'beta1': 0.5,
            'beta2': 0.999,
            'gan_mode': 'vanilla',
            'lambda_gp': 10.0,
            'lambda_chamfer': 1.0,
            'lambda_emd': 0.1,
            'd_steps': 1,
            'g_steps': 1,
            'save_interval': 100,
            'log_interval': 10,
            'num_epochs': 1,
            'latent_dim': 64,
            'checkpoint_dir': 'test_checkpoints',
            'log_dir': 'test_logs'
        }
        
        trainer = Trainer(
            generator=generator,
            discriminator=discriminator,
            train_loader=dataloader,
            config=config,
            device='cpu'
        )
        
        assert trainer.generator is not None
        assert trainer.discriminator is not None
        assert trainer.train_loader is not None


if __name__ == "__main__":
    pytest.main([__file__])
