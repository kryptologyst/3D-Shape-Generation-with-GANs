"""Training and Evaluation Utilities for 3D Shape Generation

This module provides training loops, loss functions, metrics, and evaluation
utilities for 3D GAN models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F


class GANLoss(nn.Module):
    """GAN loss function with optional gradient penalty.
    
    Args:
        gan_mode: Type of GAN loss ('vanilla', 'lsgan', 'wgangp')
        target_real_label: Label for real samples
        target_fake_label: Label for fake samples
        lambda_gp: Weight for gradient penalty (only for wgangp)
    """
    
    def __init__(
        self,
        gan_mode: str = 'vanilla',
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
        lambda_gp: float = 10.0
    ):
        super().__init__()
        self.gan_mode = gan_mode
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.lambda_gp = lambda_gp
        
        if gan_mode == 'vanilla':
            self.loss_fn = nn.BCELoss()
        elif gan_mode == 'lsgan':
            self.loss_fn = nn.MSELoss()
        elif gan_mode == 'wgangp':
            self.loss_fn = None  # Will use direct output
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")
    
    def __call__(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        discriminator: Optional[nn.Module] = None,
        real_data: Optional[torch.Tensor] = None,
        fake_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute GAN loss.
        
        Args:
            prediction: Discriminator output
            target_is_real: Whether target is real or fake
            discriminator: Discriminator model (needed for gradient penalty)
            real_data: Real data (needed for gradient penalty)
            fake_data: Fake data (needed for gradient penalty)
            
        Returns:
            Loss tensor
        """
        if self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
            
            # Add gradient penalty
            if discriminator is not None and real_data is not None and fake_data is not None:
                loss += self.lambda_gp * self._gradient_penalty(discriminator, real_data, fake_data)
            
            return loss
        else:
            if target_is_real:
                target = torch.full_like(prediction, self.target_real_label)
            else:
                target = torch.full_like(prediction, self.target_fake_label)
            
            return self.loss_fn(prediction, target)
    
    def _gradient_penalty(
        self,
        discriminator: nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP.
        
        Args:
            discriminator: Discriminator model
            real_data: Real data
            fake_data: Fake data
            
        Returns:
            Gradient penalty tensor
        """
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Random interpolation between real and fake data
        alpha = torch.rand(batch_size, 1, device=device)
        if real_data.dim() > 2:
            alpha = alpha.view(batch_size, 1, 1, 1)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Compute discriminator output for interpolated data
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=tuple(range(1, gradients.dim()))) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        
        return gradient_penalty


class ChamferDistance(nn.Module):
    """Chamfer distance loss for point clouds.
    
    Computes the bidirectional Chamfer distance between two point clouds.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Chamfer distance.
        
        Args:
            pred: Predicted point cloud (batch_size, num_points, 3)
            target: Target point cloud (batch_size, num_points, 3)
            
        Returns:
            Chamfer distance tensor
        """
        # Compute pairwise distances
        dist1 = torch.cdist(pred, target, p=2)  # (batch_size, num_points, num_points)
        dist2 = torch.cdist(target, pred, p=2)  # (batch_size, num_points, num_points)
        
        # Compute minimum distances
        min_dist1, _ = torch.min(dist1, dim=2)  # (batch_size, num_points)
        min_dist2, _ = torch.min(dist2, dim=2)  # (batch_size, num_points)
        
        # Compute Chamfer distance
        chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
        
        return chamfer_dist


class EarthMoverDistance(nn.Module):
    """Earth Mover's Distance (EMD) loss for point clouds.
    
    Computes the EMD between two point clouds using the Hungarian algorithm.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Earth Mover's Distance.
        
        Args:
            pred: Predicted point cloud (batch_size, num_points, 3)
            target: Target point cloud (batch_size, num_points, 3)
            
        Returns:
            EMD tensor
        """
        batch_size = pred.size(0)
        emd_loss = 0.0
        
        for i in range(batch_size):
            pred_i = pred[i]  # (num_points, 3)
            target_i = target[i]  # (num_points, 3)
            
            # Compute pairwise distances
            dist_matrix = torch.cdist(pred_i, target_i, p=2)  # (num_points, num_points)
            
            # Use Hungarian algorithm to find optimal assignment
            # For simplicity, we'll use a greedy approach here
            # In practice, you'd use a proper Hungarian algorithm implementation
            min_dist, _ = torch.min(dist_matrix, dim=1)
            emd_loss += torch.mean(min_dist)
        
        return emd_loss / batch_size


class Trainer:
    """Trainer class for 3D GAN models.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = 'cuda'
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Default configuration
        self.config = {
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
            'save_interval': 1000,
            'log_interval': 100,
            'num_epochs': 100,
            'latent_dim': 128,
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs'
        }
        
        if config:
            self.config.update(config)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config['lr_g'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['lr_d'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        
        # Initialize loss functions
        self.gan_loss = GANLoss(
            gan_mode=self.config['gan_mode'],
            lambda_gp=self.config['lambda_gp']
        )
        self.chamfer_loss = ChamferDistance()
        self.emd_loss = EarthMoverDistance()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'chamfer_loss': [],
            'emd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }
        
        # Create directories
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['log_dir']).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'chamfer_loss': 0.0,
            'emd_loss': 0.0,
            'd_real_acc': 0.0,
            'd_fake_acc': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, real_data in enumerate(pbar):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Generate fake data
                noise = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
                fake_data = self.generator(noise)
                
                # Train discriminator
                d_loss = self._train_discriminator(real_data, fake_data)
                
                # Train generator
                g_loss = self._train_generator(real_data, fake_data, noise)
                
                # Compute metrics
                with torch.no_grad():
                    d_real_pred = self.discriminator(real_data)
                    d_fake_pred = self.discriminator(fake_data)
                    
                    d_real_acc = (d_real_pred > 0.5).float().mean()
                    d_fake_acc = (d_fake_pred < 0.5).float().mean()
                    
                    chamfer_loss = self.chamfer_loss(fake_data, real_data)
                    emd_loss = self.emd_loss(fake_data, real_data)
                
                # Update metrics
                epoch_metrics['g_loss'] += g_loss
                epoch_metrics['d_loss'] += d_loss
                epoch_metrics['chamfer_loss'] += chamfer_loss.item()
                epoch_metrics['emd_loss'] += emd_loss.item()
                epoch_metrics['d_real_acc'] += d_real_acc.item()
                epoch_metrics['d_fake_acc'] += d_fake_acc.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'G_Loss': f'{g_loss:.4f}',
                    'D_Loss': f'{d_loss:.4f}',
                    'D_Real_Acc': f'{d_real_acc:.4f}',
                    'D_Fake_Acc': f'{d_fake_acc:.4f}'
                })
                
                # Log and save
                if batch_idx % self.config['log_interval'] == 0:
                    self._log_metrics(epoch, batch_idx, epoch_metrics)
                
                if batch_idx % self.config['save_interval'] == 0:
                    self._save_checkpoint(epoch, batch_idx)
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _train_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> float:
        """Train discriminator for one step.
        
        Args:
            real_data: Real data batch
            fake_data: Fake data batch
            
        Returns:
            Discriminator loss
        """
        self.optimizer_d.zero_grad()
        
        # Real data
        d_real_pred = self.discriminator(real_data)
        d_real_loss = self.gan_loss(d_real_pred, True)
        
        # Fake data
        d_fake_pred = self.discriminator(fake_data.detach())
        d_fake_loss = self.gan_loss(d_fake_pred, False)
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        
        d_loss.backward()
        self.optimizer_d.step()
        
        return d_loss.item()
    
    def _train_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor, noise: torch.Tensor) -> float:
        """Train generator for one step.
        
        Args:
            real_data: Real data batch
            fake_data: Fake data batch
            noise: Noise vector
            
        Returns:
            Generator loss
        """
        self.optimizer_g.zero_grad()
        
        # GAN loss
        d_fake_pred = self.discriminator(fake_data)
        g_loss = self.gan_loss(d_fake_pred, True)
        
        # Additional losses
        chamfer_loss = self.chamfer_loss(fake_data, real_data)
        emd_loss = self.emd_loss(fake_data, real_data)
        
        # Total generator loss
        total_loss = g_loss + self.config['lambda_chamfer'] * chamfer_loss + self.config['lambda_emd'] * emd_loss
        
        total_loss.backward()
        self.optimizer_g.step()
        
        return total_loss.item()
    
    def _log_metrics(self, epoch: int, batch_idx: int, metrics: Dict[str, float]) -> None:
        """Log training metrics.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            metrics: Training metrics
        """
        log_str = f'Epoch {epoch}, Batch {batch_idx}: '
        log_str += f'G_Loss: {metrics["g_loss"]:.4f}, '
        log_str += f'D_Loss: {metrics["d_loss"]:.4f}, '
        log_str += f'Chamfer: {metrics["chamfer_loss"]:.4f}, '
        log_str += f'EMD: {metrics["emd_loss"]:.4f}, '
        log_str += f'D_Real_Acc: {metrics["d_real_acc"]:.4f}, '
        log_str += f'D_Fake_Acc: {metrics["d_fake_acc"]:.4f}'
        
        print(log_str)
        
        # Save to file
        log_file = Path(self.config['log_dir']) / 'training.log'
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')
    
    def _save_checkpoint(self, epoch: int, batch_idx: int) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
        """
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        checkpoint_path = Path(self.config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save the latest checkpoint
        latest_path = Path(self.config['checkpoint_dir']) / 'latest.pth'
        torch.save(checkpoint, latest_path)
    
    def train(self) -> None:
        """Train the model for the specified number of epochs."""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_metrics = self.train_epoch(epoch)
            
            # Update history
            for key, value in epoch_metrics.items():
                self.history[key].append(value)
            
            # Validation
            if self.val_loader is not None:
                val_metrics = self.validate(epoch)
                print(f'Epoch {epoch} Validation: {val_metrics}')
            
            # Save final checkpoint
            if epoch == self.config['num_epochs']:
                self._save_checkpoint(epoch, len(self.train_loader))
        
        print("Training completed!")
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'chamfer_loss': 0.0,
            'emd_loss': 0.0,
            'd_real_acc': 0.0,
            'd_fake_acc': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for real_data in self.val_loader:
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Generate fake data
                noise = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
                fake_data = self.generator(noise)
                
                # Compute losses
                d_real_pred = self.discriminator(real_data)
                d_fake_pred = self.discriminator(fake_data)
                
                d_real_loss = self.gan_loss(d_real_pred, True)
                d_fake_loss = self.gan_loss(d_fake_pred, False)
                d_loss = d_real_loss + d_fake_loss
                
                g_loss = self.gan_loss(d_fake_pred, True)
                
                # Compute metrics
                d_real_acc = (d_real_pred > 0.5).float().mean()
                d_fake_acc = (d_fake_pred < 0.5).float().mean()
                
                chamfer_loss = self.chamfer_loss(fake_data, real_data)
                emd_loss = self.emd_loss(fake_data, real_data)
                
                # Update metrics
                val_metrics['g_loss'] += g_loss.item()
                val_metrics['d_loss'] += d_loss.item()
                val_metrics['chamfer_loss'] += chamfer_loss.item()
                val_metrics['emd_loss'] += emd_loss.item()
                val_metrics['d_real_acc'] += d_real_acc.item()
                val_metrics['d_fake_acc'] += d_fake_acc.item()
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics


def load_checkpoint(checkpoint_path: str, generator: nn.Module, discriminator: nn.Module) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        generator: Generator model
        discriminator: Discriminator model
        
    Returns:
        Checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    return checkpoint


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['g_loss', 'd_loss', 'chamfer_loss', 'emd_loss', 'd_real_acc', 'd_fake_acc']
    
    for i, metric in enumerate(metrics):
        if metric in history:
            axes[i].plot(history[metric])
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Value')
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
