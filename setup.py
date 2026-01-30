"""Setup script for the 3D Shape Generation package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="3d-shape-generation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modern implementation of 3D shape generation using GANs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/3d-shape-generation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "demo": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "3d-shape-train=scripts.train:main",
            "3d-shape-eval=scripts.evaluate:main",
            "3d-shape-demo=demo.app:main",
        ],
    },
)
