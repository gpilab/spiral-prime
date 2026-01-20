# Example Data for Spiral-PRIME

This directory contains sample data and documentation for testing and demonstrating the Spiral-PRIME reconstruction pipeline.

## Data Format Specifications

### K-space Data

**File format:** HDF5 (`.h5`) or NumPy (`.npy`)

**Required fields (for HDF5):**
- `kspace`: Complex-valued array [n_coils, n_points] or [n_coils, n_readout, n_interleaves, n_slices]
- `trajectory`: Real-valued array [n_points, n_dims] normalized to [-0.5, 0.5]

**Optional fields:**
- `coil_maps`: Coil sensitivity maps [n_coils, nz, ny, nx]
- `density_comp`: Density compensation weights [n_points]
- `metadata`: Dictionary with acquisition parameters (TR, TE, FOV, etc.)

### Example HDF5 Structure

```python
import h5py

with h5py.File('spiral_data.h5', 'w') as f:
    f.create_dataset('kspace', data=kspace_data)
    f.create_dataset('trajectory', data=trajectory)
    
    # Metadata
    meta = f.create_group('metadata')
    meta.attrs['TR'] = 2000  # ms
    meta.attrs['TE'] = 30    # ms
    meta.attrs['FOV'] = [24.0, 24.0, 16.0]  # cm
    meta.attrs['n_interleaves'] = 8
```

### Loading Data in Python

```python
import h5py
import numpy as np

# Load from HDF5
with h5py.File('spiral_data.h5', 'r') as f:
    kspace = f['kspace'][:]
    trajectory = f['trajectory'][:]
    TR = f['metadata'].attrs['TR']

# Load from NumPy
data = np.load('spiral_data.npy', allow_pickle=True).item()
kspace = data['kspace']
trajectory = data['trajectory']
```

## Sample Data

### Creating Synthetic Data

Use the provided notebook to generate synthetic data:

```python
from spiral_prime_gpi.core import spiral_physics
import numpy as np

# Create phantom
from examples.notebooks import create_shepp_logan_phantom
phantom = create_shepp_logan_phantom(128)

# Design trajectory
trajectory, _, _ = spiral_physics.design_spiral_trajectory(
    fov=24.0, resolution=0.2, n_interleaves=8
)

# Create coil maps and simulate acquisition
# (See 01_quick_start.ipynb for complete example)
```

### Downloading Real Data

Real spiral MRI data can be obtained from:

1. **mridata.org** - Public repository of MRI data
   - Search for "spiral" acquisitions
   - Download in ISMRMRD or RAW format

2. **fastMRI Dataset** - NYU fastMRI initiative
   - Contains knee and brain scans
   - Some data includes spiral trajectories

3. **Stanford CNI** - Center for Neuroimaging
   - Research datasets with spiral acquisitions

## Data Conversion

### DICOM to Spiral-PRIME Format

```python
import pydicom
import numpy as np
import h5py

# Read DICOM files
dicom_files = ['image_001.dcm', 'image_002.dcm', ...]
images = [pydicom.dcmread(f).pixel_array for f in dicom_files]

# Convert to k-space (if needed)
# ... conversion code ...

# Save in Spiral-PRIME format
with h5py.File('converted_data.h5', 'w') as f:
    f.create_dataset('kspace', data=kspace_data)
    f.create_dataset('trajectory', data=trajectory)
```

### ISMRMRD to Spiral-PRIME Format

```python
import ismrmrd
import h5py

# Read ISMRMRD file
dset = ismrmrd.Dataset('rawdata.h5', 'dataset')

# Extract k-space data
kspace_list = []
for acq in dset.acquisitions:
    kspace_list.append(acq.data)

kspace = np.array(kspace_list)

# Extract trajectory
traj_list = []
for acq in dset.acquisitions:
    traj_list.append(acq.traj)

trajectory = np.array(traj_list)

# Save
with h5py.File('converted_spiral.h5', 'w') as f:
    f.create_dataset('kspace', data=kspace)
    f.create_dataset('trajectory', data=trajectory)
```

## Validation

Before using data with Spiral-PRIME, validate it:

```python
from spiral_prime_gpi.utils import validators

# Validate k-space data
validators.validate_kspace_data(kspace)

# Validate trajectory
validators.validate_trajectory(trajectory, expected_dims=2)

# Check consistency
validators.check_data_consistency(kspace, trajectory)
```

## Notes

- Always normalize trajectory to [-0.5, 0.5] range
- Ensure k-space data is complex-valued
- Include metadata for proper reconstruction
- Check data dimensions match between k-space and trajectory

## Support

For questions about data formats:
- Open an issue on GitHub
- Consult the [documentation](../../docs/)
- See example notebooks for reference implementations
