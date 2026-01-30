# 3D Shape Generation - Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd 3D_Shape_Generation

# Install dependencies
pip install -r requirements.txt
```

## Quick Demo

### 1. Train a Model

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/default.yaml --device cuda
```

### 2. Evaluate the Model

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/latest.pth

# Generate samples
python scripts/evaluate.py --checkpoint checkpoints/latest.pth --save-samples
```

### 3. Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Python API

```python
import torch
from src.models.gan_models import create_model
from src.data.datasets import SyntheticDataset
from src.train.trainer import Trainer

# Create models
generator, discriminator = create_model(
    model_type="point_cloud",
    latent_dim=128,
    num_points=2048
)

# Create dataset
dataset = SyntheticDataset(num_samples=1000, num_points=2048)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train
trainer = Trainer(generator, discriminator, dataloader)
trainer.train()

# Generate samples
generator.eval()
with torch.no_grad():
    noise = torch.randn(4, 128)
    generated_shapes = generator(noise)
```

## Configuration

Edit `configs/default.yaml` to customize:

- Model architecture
- Training parameters
- Data settings
- Evaluation metrics

## Visualization

```python
from src.utils.visualization import PointCloudVisualizer

visualizer = PointCloudVisualizer()
visualizer.visualize_matplotlib(points, title="My Point Cloud")
```

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Project Structure

```
src/
├── models/          # GAN architectures
├── data/            # Data loading
├── train/           # Training utilities
├── utils/           # Visualization tools
configs/             # Configuration files
scripts/             # Training/evaluation scripts
demo/                # Interactive demo
tests/               # Unit tests
notebooks/           # Jupyter notebooks
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config
2. **Import errors**: Ensure all dependencies are installed
3. **Visualization issues**: Check matplotlib/plotly installation

### Getting Help

- Check the README.md for detailed documentation
- Run tests to verify installation
- Check logs/ directory for training logs
