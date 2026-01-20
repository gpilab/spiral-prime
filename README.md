# Spiral-PRIME

**Physics-based Reconstruction with Iterative Model-based Enhancement**

Spiral-PRIME provides a unified 3D reconstruction pipeline implemented as custom nodes for the [Graphical Programming Interface (GPI)](http://gpilab.com/) framework. This framework is designed to reconstruct high-resolution, multi-contrast brain volumes from a single steady-state localized-quadratic (LQ) spin-echo acquisition.

## Features

- **Unified 3D Reconstruction Pipeline**: Comprehensive workflow from raw k-space data to reconstructed volumes
- **GPI Integration**: Custom nodes designed for the GPI framework for intuitive workflow design
- **Multi-Contrast Imaging**: Reconstruct multiple tissue contrasts from a single acquisition
- **Advanced Physics Modeling**: Incorporates steady-state LQ spin-echo physics
- **Iterative Reconstruction**: Model-based enhancement with regularization (TV, L2)
- **Efficient Processing**: Optimized with Numba JIT compilation for fast reconstruction

## Installation

### Prerequisites

- Python 3.7 or higher
- [GPI Framework](http://gpilab.com/) (optional, for GUI workflow design)

### Install from source

```bash
git clone https://github.com/gpilab/spiral-prime.git
cd spiral-prime
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

### Development installation

```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Using GPI Nodes

1. Launch GPI:
   ```bash
   gpi
   ```

2. Add Spiral-PRIME nodes from the node menu:
   - `Spiral_Prime_Reconstruction`: Main reconstruction node
   - `Coil_Sensitivity_Map`: Coil sensitivity estimation
   - `Spiral_Trajectory`: Spiral trajectory design/loading

3. Connect nodes to create your reconstruction workflow

### Using Python API

```python
import numpy as np
from spiral_prime_gpi.core import spiral_physics, mri_math
from spiral_prime_gpi.core.optimization import iterative_sense_recon

# Load k-space data
kspace_data = ...  # Your multi-coil k-space data

# Load or design trajectory
trajectory = ...  # Spiral trajectory (k_x, k_y, k_z)

# Estimate coil sensitivity maps
coil_maps = mri_math.estimate_coil_sensitivity(kspace_data)

# Perform reconstruction
reconstructed_volume = iterative_sense_recon(
    kspace_data, 
    trajectory, 
    coil_maps,
    n_iterations=10,
    regularization='tv'
)
```

## Documentation

- [Installation Guide](docs/installation.md)
- [GPI Integration](docs/gpi_integration.md)
- [Algorithm Overview](docs/algorithm_overview.md)
- [Parameters Guide](docs/parameters_guide.md)
- [API Reference](docs/api_reference.md)

## Project Structure

```
spiral-prime/
├── spiral_prime_gpi/        # Main Python package
│   ├── core/                # Core reconstruction algorithms
│   ├── nodes/               # GPI node implementations
│   ├── config/              # Configuration files
│   └── utils/               # Utility functions
├── gpi_nodes/               # GPI node wrappers
├── examples/                # Example workflows and notebooks
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## Examples

Check the `examples/` directory for:
- GPI network files (`.gpi`)
- Jupyter notebooks demonstrating the API
- Sample reconstruction workflows

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=spiral_prime_gpi tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use Spiral-PRIME in your research, please cite:

```
@software{spiral_prime,
  title = {Spiral-PRIME: Physics-based Reconstruction with Iterative Model-based Enhancement},
  author = {GPI Lab},
  year = {2026},
  url = {https://github.com/gpilab/spiral-prime}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- GPI Lab for the GPI framework
- Medical imaging reconstruction community

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.