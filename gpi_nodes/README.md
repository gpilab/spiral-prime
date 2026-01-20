# Spiral-PRIME GPI Nodes

This directory contains custom GPI nodes for the Spiral-PRIME reconstruction pipeline.

## Available Nodes

### 1. Spiral_Prime_Reconstruction
Main reconstruction node that performs iterative SENSE reconstruction of spiral k-space data.

**Inputs:**
- `kspace`: Multi-coil k-space data [n_coils, n_points]
- `trajectory`: k-space trajectory [n_points, n_dims] normalized to [-0.5, 0.5]
- `coil_maps` (optional): Coil sensitivity maps
- `density_comp` (optional): Density compensation weights

**Outputs:**
- `image`: Reconstructed image
- `coil_maps_out`: Coil sensitivity maps used

**Parameters:**
- Algorithm: Iterative SENSE, CG-SENSE, or ADMM
- Iterations: Number of iterations (1-100)
- Regularization: None, L1, L2, TV, or TV (Chambolle-Pock)
- Lambda: Regularization weight
- Coil Combine: SOS, Adaptive, or Walsh
- Image Size: Output dimensions (auto or [nz,ny,nx])

### 2. Coil_Sensitivity_Map
Estimates coil sensitivity maps from multi-coil data.

**Inputs:**
- `data`: Multi-coil k-space or image data [n_coils, ...]

**Outputs:**
- `coil_maps`: Estimated coil sensitivity maps
- `combined_image`: Combined image using estimated maps

**Parameters:**
- Method: SOS, Adaptive, or Walsh
- Smoothing: Gaussian smoothing sigma (0-20)
- Auto-compute: Automatically compute on data change

### 3. Spiral_Trajectory
Designs or loads spiral k-space trajectories.

**Inputs:**
- `trajectory_file` (optional): Path to trajectory file for loading

**Outputs:**
- `trajectory`: k-space trajectory [n_points, n_dims]
- `gradient`: Gradient waveforms [n_points, n_dims] in mT/m
- `density_comp`: Density compensation weights

**Parameters:**
- Mode: Design or Load from file
- Type: 2D Spiral or 3D Stack-of-Spirals
- FOV: Field of view in cm
- FOV Z: Through-plane FOV for 3D
- Resolution: Spatial resolution in mm
- Interleaves: Number of spiral interleaves (1-64)
- Max Gradient: Maximum gradient amplitude in mT/m
- Max Slew: Maximum slew rate in T/m/s
- Density Comp: Method for density compensation

## Installation

1. Make sure GPI is installed and working
2. Install Spiral-PRIME:
   ```bash
   cd /path/to/spiral-prime
   pip install -e .
   ```

3. Add this directory to your GPI node path:
   - Open GPI
   - Go to Config → Node Path
   - Add the path to this `gpi_nodes` directory
   - Restart GPI

4. The nodes should now appear in the GPI node menu under "Spiral-PRIME"

## Usage Example

### Basic Reconstruction Workflow

1. **Load or Generate Trajectory:**
   - Add `Spiral_Trajectory` node
   - Configure FOV and resolution
   - Click "Generate"

2. **Load K-space Data:**
   - Add a data loading node (e.g., `ReadHDF5`)
   - Load your multi-coil k-space data

3. **Estimate Coil Maps (optional):**
   - Add `Coil_Sensitivity_Map` node
   - Connect k-space data
   - Click "Estimate"

4. **Reconstruct:**
   - Add `Spiral_Prime_Reconstruction` node
   - Connect k-space, trajectory, and coil maps
   - Select algorithm and parameters
   - Click "Reconstruct"

5. **View Results:**
   - Add visualization nodes (e.g., `ImageDisplay`)
   - Connect reconstructed image

### Example Network Layout

```
[Spiral_Trajectory] → trajectory ─┐
                                   │
[ReadHDF5] → kspace ───────────────┼→ [Spiral_Prime_Reconstruction] → image → [ImageDisplay]
                                   │
[Coil_Sensitivity_Map] → maps ────┘
```

## Data Format Requirements

### K-space Data
- Complex-valued numpy array
- Shape: [n_coils, n_points] or [n_coils, n_readout, n_interleaves, n_slices]
- Dtype: complex64 or complex128

### Trajectory
- Real-valued numpy array
- Shape: [n_points, n_dims] where n_dims = 2 or 3
- Normalized to [-0.5, 0.5]
- Dtype: float32 or float64

### Coil Maps
- Complex-valued numpy array
- Shape: [n_coils, nz, ny, nx] or [n_coils, ny, nx]
- Dtype: complex64 or complex128

## Troubleshooting

**Node doesn't appear in GPI:**
- Check that spiral-prime package is installed
- Verify GPI node path includes this directory
- Restart GPI after adding node path

**Reconstruction errors:**
- Verify k-space and trajectory dimensions match
- Check that trajectory is normalized to [-0.5, 0.5]
- Ensure k-space data is complex-valued
- Try reducing number of iterations or regularization weight

**Memory errors:**
- Reduce image size
- Use fewer iterations
- Enable GPU acceleration if available
- Process fewer coils at once

## Advanced Features

### Custom Configuration
Save and load reconstruction parameters using the "Load Config" and "Save Config" buttons in the reconstruction node.

### Multi-Contrast Reconstruction
For multi-contrast acquisitions, process each contrast separately or use the multi-contrast reconstruction features in the core library.

### GPU Acceleration
If GPU is available and CUDA is installed, reconstruction can be accelerated by enabling GPU in the performance settings.

## Support

For issues or questions:
- Open an issue on GitHub: https://github.com/gpilab/spiral-prime
- Check documentation: https://github.com/gpilab/spiral-prime/docs

## Citation

If you use these nodes in your research, please cite:
```
@software{spiral_prime,
  title = {Spiral-PRIME: Physics-based Reconstruction with Iterative Model-based Enhancement},
  author = {GPI Lab},
  year = {2026},
  url = {https://github.com/gpilab/spiral-prime}
}
```
