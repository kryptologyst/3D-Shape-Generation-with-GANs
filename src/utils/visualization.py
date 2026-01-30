"""3D Visualization Utilities

This module provides visualization utilities for 3D shapes including
point clouds and voxel grids using Open3D and matplotlib.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import List, Tuple, Optional, Union, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import trimesh
from pathlib import Path


class PointCloudVisualizer:
    """Visualizer for 3D point clouds."""
    
    def __init__(self, point_size: float = 1.0, background_color: str = 'white'):
        """Initialize the visualizer.
        
        Args:
            point_size: Size of points in the visualization
            background_color: Background color of the visualization
        """
        self.point_size = point_size
        self.background_color = background_color
    
    def visualize_open3d(
        self,
        points: Union[np.ndarray, torch.Tensor],
        colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
        title: str = "Point Cloud",
        save_path: Optional[str] = None
    ) -> None:
        """Visualize point cloud using Open3D.
        
        Args:
            points: Point cloud data (N, 3)
            colors: Point colors (N, 3) or (3,) for uniform color
            title: Title for the visualization
            save_path: Path to save the visualization
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Set colors
        if colors is not None:
            if colors.ndim == 1:
                # Uniform color
                pcd.paint_uniform_color(colors)
            else:
                # Per-point colors
                pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Default color based on z-coordinate
            z_coords = points[:, 2]
            colors = plt.cm.viridis((z_coords - z_coords.min()) / (z_coords.max() - z_coords.min()))[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd], window_name=title)
        
        # Save if requested
        if save_path:
            o3d.io.write_point_cloud(save_path, pcd)
    
    def visualize_matplotlib(
        self,
        points: Union[np.ndarray, torch.Tensor],
        colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
        title: str = "Point Cloud",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Visualize point cloud using matplotlib.
        
        Args:
            points: Point cloud data (N, 3)
            colors: Point colors (N, 3) or (3,) for uniform color
            title: Title for the visualization
            save_path: Path to save the visualization
            figsize: Figure size
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set colors
        if colors is not None:
            if colors.ndim == 1:
                # Uniform color
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=colors, s=self.point_size)
            else:
                # Per-point colors
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=colors, s=self.point_size)
        else:
            # Default color based on z-coordinate
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 2], cmap='viridis', s=self.point_size)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                             points[:, 1].max() - points[:, 1].min(),
                             points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_plotly(
        self,
        points: Union[np.ndarray, torch.Tensor],
        colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
        title: str = "Point Cloud",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Visualize point cloud using Plotly.
        
        Args:
            points: Point cloud data (N, 3)
            colors: Point colors (N, 3) or (3,) for uniform color
            title: Title for the visualization
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        
        # Set colors
        if colors is not None:
            if colors.ndim == 1:
                # Uniform color
                color_str = f'rgb({colors[0]}, {colors[1]}, {colors[2]})'
                scatter_color = color_str
            else:
                # Per-point colors
                scatter_color = colors
        else:
            # Default color based on z-coordinate
            scatter_color = points[:, 2]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=self.point_size,
                color=scatter_color,
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Point Cloud'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_comparison(
        self,
        real_points: Union[np.ndarray, torch.Tensor],
        fake_points: Union[np.ndarray, torch.Tensor],
        title: str = "Real vs Generated Point Clouds",
        save_path: Optional[str] = None
    ) -> None:
        """Visualize comparison between real and generated point clouds.
        
        Args:
            real_points: Real point cloud data (N, 3)
            fake_points: Generated point cloud data (N, 3)
            title: Title for the visualization
            save_path: Path to save the visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': '3d'})
        
        # Real points
        if isinstance(real_points, torch.Tensor):
            real_points = real_points.detach().cpu().numpy()
        
        ax1.scatter(real_points[:, 0], real_points[:, 1], real_points[:, 2], 
                   c=real_points[:, 2], cmap='viridis', s=self.point_size)
        ax1.set_title('Real Point Cloud')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Generated points
        if isinstance(fake_points, torch.Tensor):
            fake_points = fake_points.detach().cpu().numpy()
        
        ax2.scatter(fake_points[:, 0], fake_points[:, 1], fake_points[:, 2], 
                   c=fake_points[:, 2], cmap='viridis', s=self.point_size)
        ax2.set_title('Generated Point Cloud')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class VoxelVisualizer:
    """Visualizer for 3D voxel grids."""
    
    def __init__(self, voxel_size: float = 1.0, opacity: float = 0.8):
        """Initialize the visualizer.
        
        Args:
            voxel_size: Size of voxels in the visualization
            opacity: Opacity of voxels
        """
        self.voxel_size = voxel_size
        self.opacity = opacity
    
    def visualize_matplotlib(
        self,
        voxels: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5,
        title: str = "Voxel Grid",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Visualize voxel grid using matplotlib.
        
        Args:
            voxels: Voxel grid data (H, W, D)
            threshold: Threshold for voxel visibility
            title: Title for the visualization
            save_path: Path to save the visualization
            figsize: Figure size
        """
        if isinstance(voxels, torch.Tensor):
            voxels = voxels.detach().cpu().numpy()
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get voxel positions
        filled = voxels > threshold
        positions = np.where(filled)
        
        if len(positions[0]) > 0:
            ax.scatter(positions[0], positions[1], positions[2], 
                      c=voxels[filled], cmap='viridis', s=self.voxel_size)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_plotly(
        self,
        voxels: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5,
        title: str = "Voxel Grid",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Visualize voxel grid using Plotly.
        
        Args:
            voxels: Voxel grid data (H, W, D)
            threshold: Threshold for voxel visibility
            title: Title for the visualization
            save_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if isinstance(voxels, torch.Tensor):
            voxels = voxels.detach().cpu().numpy()
        
        # Get voxel positions
        filled = voxels > threshold
        positions = np.where(filled)
        
        if len(positions[0]) == 0:
            # Empty voxel grid
            fig = go.Figure()
            fig.update_layout(title=title)
            return fig
        
        fig = go.Figure(data=[go.Scatter3d(
            x=positions[0],
            y=positions[1],
            z=positions[2],
            mode='markers',
            marker=dict(
                size=self.voxel_size,
                color=voxels[filled],
                colorscale='Viridis',
                opacity=self.opacity
            ),
            name='Voxels'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_mesh(
        self,
        voxels: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5,
        title: str = "Voxel Mesh",
        save_path: Optional[str] = None
    ) -> None:
        """Visualize voxel grid as mesh using Open3D.
        
        Args:
            voxels: Voxel grid data (H, W, D)
            threshold: Threshold for voxel visibility
            title: Title for the visualization
            save_path: Path to save the visualization
        """
        if isinstance(voxels, torch.Tensor):
            voxels = voxels.detach().cpu().numpy()
        
        # Convert voxels to mesh
        mesh = self._voxels_to_mesh(voxels, threshold)
        
        # Visualize
        o3d.visualization.draw_geometries([mesh], window_name=title)
        
        # Save if requested
        if save_path:
            o3d.io.write_triangle_mesh(save_path, mesh)
    
    def _voxels_to_mesh(self, voxels: np.ndarray, threshold: float = 0.5) -> o3d.geometry.TriangleMesh:
        """Convert voxel grid to triangle mesh.
        
        Args:
            voxels: Voxel grid data (H, W, D)
            threshold: Threshold for voxel visibility
            
        Returns:
            Triangle mesh
        """
        # Create voxel grid
        voxel_grid = o3d.geometry.VoxelGrid()
        
        # Convert to binary voxels
        binary_voxels = (voxels > threshold).astype(np.uint8)
        
        # Create mesh from voxels
        mesh = o3d.geometry.TriangleMesh()
        
        # Simple approach: create cubes for each voxel
        for i in range(binary_voxels.shape[0]):
            for j in range(binary_voxels.shape[1]):
                for k in range(binary_voxels.shape[2]):
                    if binary_voxels[i, j, k] == 1:
                        # Create cube mesh
                        cube = o3d.geometry.TriangleMesh.create_box(
                            width=1, height=1, depth=1
                        )
                        cube.translate([i, j, k])
                        mesh += cube
        
        return mesh


class TrainingVisualizer:
    """Visualizer for training progress and results."""
    
    def __init__(self, save_dir: str = "assets"):
        """Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot training curves.
        
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
        else:
            plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_generation_grid(
        self,
        generator: torch.nn.Module,
        num_samples: int = 16,
        latent_dim: int = 128,
        device: str = 'cuda',
        save_path: Optional[str] = None
    ) -> None:
        """Create a grid of generated samples.
        
        Args:
            generator: Trained generator model
            num_samples: Number of samples to generate
            latent_dim: Dimension of latent vector
            device: Device to run on
            save_path: Path to save the visualization
        """
        generator.eval()
        
        with torch.no_grad():
            # Generate samples
            noise = torch.randn(num_samples, latent_dim, device=device)
            generated_samples = generator(noise)
            
            # Create grid
            grid_size = int(np.ceil(np.sqrt(num_samples)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15), 
                                   subplot_kw={'projection': '3d'})
            
            if grid_size == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i in range(num_samples):
                points = generated_samples[i].cpu().numpy()
                
                axes[i].scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=points[:, 2], cmap='viridis', s=1)
                axes[i].set_title(f'Sample {i+1}')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')
                axes[i].set_zlabel('Z')
            
            # Hide unused subplots
            for i in range(num_samples, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Generated 3D Shapes')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.save_dir / 'generation_grid.png', dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def create_interpolation_visualization(
        self,
        generator: torch.nn.Module,
        latent_dim: int = 128,
        num_steps: int = 10,
        device: str = 'cuda',
        save_path: Optional[str] = None
    ) -> None:
        """Create interpolation visualization between two latent vectors.
        
        Args:
            generator: Trained generator model
            latent_dim: Dimension of latent vector
            num_steps: Number of interpolation steps
            device: Device to run on
            save_path: Path to save the visualization
        """
        generator.eval()
        
        with torch.no_grad():
            # Generate two random latent vectors
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)
            
            # Create interpolation
            interpolations = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                generated = generator(z_interp)
                interpolations.append(generated[0].cpu().numpy())
            
            # Create visualization
            fig, axes = plt.subplots(2, num_steps // 2, figsize=(20, 8), 
                                   subplot_kw={'projection': '3d'})
            axes = axes.flatten()
            
            for i, points in enumerate(interpolations):
                axes[i].scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=points[:, 2], cmap='viridis', s=1)
                axes[i].set_title(f'Step {i+1}')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')
                axes[i].set_zlabel('Z')
            
            plt.suptitle('Latent Space Interpolation')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.save_dir / 'interpolation.png', dpi=300, bbox_inches='tight')
            
            plt.show()


def save_point_cloud(points: Union[np.ndarray, torch.Tensor], file_path: str) -> None:
    """Save point cloud to file.
    
    Args:
        points: Point cloud data (N, 3)
        file_path: Path to save the file
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    o3d.io.write_point_cloud(file_path, pcd)


def load_point_cloud(file_path: str) -> np.ndarray:
    """Load point cloud from file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Point cloud data (N, 3)
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)
