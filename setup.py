"""Setup script for Spiral-PRIME package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="spiral-prime",
    version="0.1.0",
    author="GPI Lab",
    author_email="gpilab@users.noreply.github.com",
    description="Physics-based Reconstruction with Iterative Model-based Enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gpilab/spiral-prime",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "h5py>=3.0",
        "nibabel>=3.2",
        "pyyaml>=5.4",
        "numba>=0.55",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    package_data={
        "spiral_prime_gpi": ["config/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "spiral-prime=spiral_prime_gpi.cli:main",
        ],
    },
)
