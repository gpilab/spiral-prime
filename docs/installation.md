# Installation Guide

## Prerequisites

### Required Software
- Python 3.7 or higher
- pip (Python package manager)

### Optional Software
- [GPI Framework](http://gpilab.com/) - For using the graphical interface
- CUDA Toolkit - For GPU acceleration (optional)

## Installation Methods

### Method 1: Install from Source (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gpilab/spiral-prime.git
   cd spiral-prime
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   python -c "import spiral_prime_gpi; print(spiral_prime_gpi.__version__)"
   ```

### Method 2: Install from PyPI (when available)

```bash
pip install spiral-prime
```

## Development Installation

For development work, install with development dependencies:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

This includes:
- pytest for testing
- black for code formatting
- flake8 for linting
- jupyter for notebooks
- matplotlib for visualization

## GPI Integration

If you want to use the GPI nodes:

1. **Install GPI Framework:**
   Follow instructions at http://gpilab.com/

2. **Add Spiral-PRIME nodes to GPI:**
   - Launch GPI
   - Go to `Config` → `Node Path`
   - Add the path: `/path/to/spiral-prime/gpi_nodes`
   - Restart GPI

3. **Verify GPI nodes:**
   - In GPI, check the node menu
   - Look for "Spiral-PRIME" category
   - You should see the reconstruction nodes

## Verifying Installation

Run the test suite to verify everything is working:

```bash
pytest tests/
```

## Troubleshooting

### Import Errors

**Problem:** `ImportError: No module named 'spiral_prime_gpi'`

**Solution:** 
- Make sure you installed the package: `pip install -e .`
- Check that you're using the correct Python environment
- Try: `pip list | grep spiral`

### NumPy/SciPy Errors

**Problem:** Errors related to NumPy or SciPy

**Solution:**
- Update NumPy and SciPy:
  ```bash
  pip install --upgrade numpy scipy
  ```

### GPI Node Not Found

**Problem:** Spiral-PRIME nodes don't appear in GPI

**Solution:**
1. Verify package is installed: `pip show spiral-prime`
2. Check GPI node path settings
3. Restart GPI after adding node path
4. Check GPI console for error messages

### Memory Errors

**Problem:** Out of memory during reconstruction

**Solution:**
- Reduce image size
- Use fewer iterations
- Process data in chunks
- Use a machine with more RAM

### CUDA/GPU Errors

**Problem:** GPU acceleration not working

**Solution:**
- Verify CUDA is installed: `nvidia-smi`
- Check PyTorch/CuPy installation if using GPU features
- Disable GPU in configuration if not needed

## Platform-Specific Notes

### Linux
No special requirements. Standard installation should work.

### macOS
- May need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Windows
- Install Visual C++ Build Tools if compilation errors occur
- Use Anaconda Python for easier dependency management:
  ```bash
  conda create -n spiral-prime python=3.9
  conda activate spiral-prime
  pip install -e .
  ```

## Updating

To update to the latest version:

```bash
cd spiral-prime
git pull
pip install -e . --upgrade
```

## Uninstallation

```bash
pip uninstall spiral-prime
```

## Next Steps

After installation:
1. Read the [Quick Start Guide](../README.md#quick-start)
2. Try the [example notebooks](../examples/notebooks/)
3. Read the [API Reference](api_reference.md)
4. Check out [GPI Integration Guide](gpi_integration.md)

## Getting Help

- GitHub Issues: https://github.com/gpilab/spiral-prime/issues
- Documentation: https://github.com/gpilab/spiral-prime/docs
- GPI Forum: http://gpilab.com/forum
