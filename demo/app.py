"""Streamlit Demo App for 3D Shape Generation

This module provides an interactive Streamlit demo for 3D shape generation
using trained GAN models.
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gan_models import create_model, initialize_weights
from data.datasets import SyntheticDataset
from utils.visualization import PointCloudVisualizer, TrainingVisualizer
from train.trainer import load_checkpoint


class ShapeGenerationDemo:
    """Demo application for 3D shape generation."""
    
    def __init__(self):
        """Initialize the demo."""
        self.device = self._get_device()
        self.generator = None
        self.discriminator = None
        self.latent_dim = 128
        self.num_points = 2048
        
        # Initialize visualizers
        self.pc_visualizer = PointCloudVisualizer()
        self.training_visualizer = TrainingVisualizer()
        
        # Load synthetic dataset for demonstration
        self.synthetic_dataset = SyntheticDataset(
            num_samples=100,
            shape_types=['sphere', 'cube', 'cylinder'],
            num_points=self.num_points,
            data_type='point_cloud'
        )
    
    def _get_device(self) -> str:
        """Get the best available device.
        
        Returns:
            Device string
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_model(self, model_type: str = "point_cloud", checkpoint_path: Optional[str] = None):
        """Load a trained model.
        
        Args:
            model_type: Type of model to load
            checkpoint_path: Path to checkpoint file
        """
        try:
            # Create models
            self.generator, self.discriminator = create_model(
                model_type=model_type,
                latent_dim=self.latent_dim,
                num_points=self.num_points
            )
            
            # Initialize weights
            initialize_weights(self.generator)
            initialize_weights(self.discriminator)
            
            # Load checkpoint if provided
            if checkpoint_path and Path(checkpoint_path).exists():
                checkpoint = load_checkpoint(checkpoint_path, self.generator, self.discriminator)
                st.success(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            else:
                st.warning("No checkpoint provided, using randomly initialized weights")
            
            # Move to device
            self.generator = self.generator.to(self.device)
            self.discriminator = self.discriminator.to(self.device)
            
            self.generator.eval()
            self.discriminator.eval()
            
            st.success(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    def generate_shapes(self, num_shapes: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """Generate 3D shapes.
        
        Args:
            num_shapes: Number of shapes to generate
            seed: Random seed for reproducibility
            
        Returns:
            Generated point clouds
        """
        if self.generator is None:
            st.error("Please load a model first")
            return None
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        with torch.no_grad():
            noise = torch.randn(num_shapes, self.latent_dim, device=self.device)
            generated_shapes = self.generator(noise)
        
        return generated_shapes
    
    def interpolate_shapes(self, num_steps: int = 10, seed: Optional[int] = None) -> List[torch.Tensor]:
        """Interpolate between two random shapes.
        
        Args:
            num_steps: Number of interpolation steps
            seed: Random seed for reproducibility
            
        Returns:
            List of interpolated point clouds
        """
        if self.generator is None:
            st.error("Please load a model first")
            return []
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        with torch.no_grad():
            # Generate two random latent vectors
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)
            
            interpolations = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                generated = self.generator(z_interp)
                interpolations.append(generated[0])
        
        return interpolations
    
    def visualize_point_cloud(self, points: torch.Tensor, title: str = "Point Cloud") -> go.Figure:
        """Visualize a point cloud using Plotly.
        
        Args:
            points: Point cloud data (N, 3)
            title: Title for the visualization
            
        Returns:
            Plotly figure
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points[:, 2],
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
        
        return fig
    
    def create_comparison_plot(self, real_points: torch.Tensor, fake_points: torch.Tensor) -> go.Figure:
        """Create a comparison plot between real and generated point clouds.
        
        Args:
            real_points: Real point cloud data
            fake_points: Generated point cloud data
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Real Point Cloud', 'Generated Point Cloud')
        )
        
        # Real points
        if isinstance(real_points, torch.Tensor):
            real_points = real_points.detach().cpu().numpy()
        
        fig.add_trace(
            go.Scatter3d(
                x=real_points[:, 0],
                y=real_points[:, 1],
                z=real_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=real_points[:, 2], colorscale='Viridis', opacity=0.8),
                name='Real'
            ),
            row=1, col=1
        )
        
        # Generated points
        if isinstance(fake_points, torch.Tensor):
            fake_points = fake_points.detach().cpu().numpy()
        
        fig.add_trace(
            go.Scatter3d(
                x=fake_points[:, 0],
                y=fake_points[:, 1],
                z=fake_points[:, 2],
                mode='markers',
                marker=dict(size=2, color=fake_points[:, 2], colorscale='Viridis', opacity=0.8),
                name='Generated'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Real vs Generated Point Clouds',
            width=1000,
            height=500
        )
        
        return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="3D Shape Generation Demo",
        page_icon="🎲",
        layout="wide"
    )
    
    st.title("🎲 3D Shape Generation Demo")
    st.markdown("Generate and visualize 3D shapes using Generative Adversarial Networks (GANs)")
    
    # Initialize demo
    if 'demo' not in st.session_state:
        st.session_state.demo = ShapeGenerationDemo()
    
    demo = st.session_state.demo
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["point_cloud", "improved_point_cloud", "voxel"],
        help="Select the type of 3D GAN model to use"
    )
    
    # Checkpoint upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Checkpoint",
        type=['pth'],
        help="Upload a trained model checkpoint (.pth file)"
    )
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            if uploaded_file is not None:
                # Save uploaded file temporarily
                checkpoint_path = f"temp_checkpoint.pth"
                with open(checkpoint_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                demo.load_model(model_type, checkpoint_path)
                os.remove(checkpoint_path)  # Clean up
            else:
                demo.load_model(model_type)
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    num_shapes = st.sidebar.slider(
        "Number of Shapes",
        min_value=1,
        max_value=16,
        value=4,
        help="Number of shapes to generate"
    )
    
    use_seed = st.sidebar.checkbox("Use Random Seed", help="Enable for reproducible results")
    seed = None
    if use_seed:
        seed = st.sidebar.number_input("Seed", value=42, help="Random seed for reproducibility")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Generate Shapes", "Interpolation", "Compare with Real", "About"])
    
    with tab1:
        st.header("Generate 3D Shapes")
        
        if st.button("Generate Shapes", type="primary"):
            if demo.generator is None:
                st.error("Please load a model first using the sidebar")
            else:
                with st.spinner("Generating shapes..."):
                    generated_shapes = demo.generate_shapes(num_shapes, seed)
                    
                    if generated_shapes is not None:
                        st.success(f"Generated {num_shapes} shapes successfully!")
                        
                        # Display shapes in a grid
                        cols = st.columns(2)
                        for i, shape in enumerate(generated_shapes):
                            with cols[i % 2]:
                                fig = demo.visualize_point_cloud(shape, f"Generated Shape {i+1}")
                                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Latent Space Interpolation")
        
        interpolation_steps = st.slider(
            "Interpolation Steps",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of steps in the interpolation"
        )
        
        if st.button("Generate Interpolation", type="primary"):
            if demo.generator is None:
                st.error("Please load a model first using the sidebar")
            else:
                with st.spinner("Generating interpolation..."):
                    interpolations = demo.interpolate_shapes(interpolation_steps, seed)
                    
                    if interpolations:
                        st.success(f"Generated interpolation with {interpolation_steps} steps!")
                        
                        # Display interpolation
                        fig = make_subplots(
                            rows=2,
                            cols=interpolation_steps // 2,
                            specs=[[{'type': 'scatter3d'} for _ in range(interpolation_steps // 2)] for _ in range(2)],
                            subplot_titles=[f"Step {i+1}" for i in range(interpolation_steps)]
                        )
                        
                        for i, points in enumerate(interpolations):
                            row = i // (interpolation_steps // 2) + 1
                            col = i % (interpolation_steps // 2) + 1
                            
                            points_np = points.detach().cpu().numpy()
                            
                            fig.add_trace(
                                go.Scatter3d(
                                    x=points_np[:, 0],
                                    y=points_np[:, 1],
                                    z=points_np[:, 2],
                                    mode='markers',
                                    marker=dict(size=1, color=points_np[:, 2], colorscale='Viridis', opacity=0.8),
                                    name=f'Step {i+1}'
                                ),
                                row=row, col=col
                            )
                        
                        fig.update_layout(
                            title='Latent Space Interpolation',
                            height=800,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Compare with Real Data")
        
        if st.button("Generate Comparison", type="primary"):
            if demo.generator is None:
                st.error("Please load a model first using the sidebar")
            else:
                with st.spinner("Generating comparison..."):
                    # Get real data
                    real_data = demo.synthetic_dataset[0]  # Get first sample
                    
                    # Generate fake data
                    fake_data = demo.generate_shapes(1, seed)
                    
                    if fake_data is not None:
                        st.success("Generated comparison successfully!")
                        
                        # Create comparison plot
                        fig = demo.create_comparison_plot(real_data, fake_data[0])
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("About This Demo")
        
        st.markdown("""
        ## 3D Shape Generation with GANs
        
        This demo showcases 3D shape generation using Generative Adversarial Networks (GANs).
        
        ### Features:
        - **Point Cloud Generation**: Generate 3D point clouds from random noise
        - **Latent Space Interpolation**: Smoothly interpolate between different shapes
        - **Real vs Generated Comparison**: Compare generated shapes with real data
        - **Interactive 3D Visualization**: Rotate and zoom 3D point clouds
        
        ### Model Types:
        - **Point Cloud GAN**: Basic MLP-based generator for point clouds
        - **Improved Point Cloud GAN**: Enhanced generator with skip connections
        - **Voxel GAN**: 3D convolutional generator for voxel grids
        
        ### Technical Details:
        - **Latent Dimension**: 128
        - **Point Cloud Size**: 2048 points
        - **Device Support**: CUDA, MPS (Apple Silicon), CPU
        - **Visualization**: Plotly 3D scatter plots
        
        ### Usage:
        1. Select a model type from the sidebar
        2. Upload a trained checkpoint (optional)
        3. Click "Load Model" to initialize the model
        4. Use the tabs to generate and visualize shapes
        
        ### Note:
        This demo uses synthetic data for demonstration purposes. In practice, you would
        train the models on real 3D shape datasets like ShapeNet, ModelNet, or custom datasets.
        """)


if __name__ == "__main__":
    main()
