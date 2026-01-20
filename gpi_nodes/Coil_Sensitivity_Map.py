"""
Coil Sensitivity Map estimation GPI node.
"""

try:
    import gpi
    HAS_GPI = True
except ImportError:
    HAS_GPI = False
    class gpi:
        @staticmethod
        def NodeAPI(x):
            return x

import numpy as np
from ...core import mri_math
from ...utils import logging_config, validators


class ExternalNode(gpi.NodeAPI):
    """
    Coil Sensitivity Map Estimation Node
    
    Estimates coil sensitivity maps from multi-coil k-space or image data.
    
    INPUT PORTS:
    - data: Multi-coil data [n_coils, ...] (k-space or image domain)
    - data_type: 'kspace' or 'image' (optional, auto-detect if not provided)
    
    OUTPUT PORTS:
    - coil_maps: Coil sensitivity maps [n_coils, ...image_shape]
    - combined_image: Combined image using estimated maps
    
    WIDGETS:
    - Method: Estimation method (SOS, Adaptive, Walsh)
    - Smoothing: Gaussian smoothing sigma
    - Auto-compute: Automatically compute on data change
    """
    
    def initUI(self):
        """Initialize the UI."""
        # Input ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64, np.complex128],
                      obligation=gpi.REQUIRED)
        
        # Output ports
        self.addOutPort('coil_maps', 'NPYarray', dtype=np.complex64)
        self.addOutPort('combined_image', 'NPYarray', dtype=np.complex64)
        
        # Widgets
        self.addWidget('PushButton', 'Estimate', toggle=False)
        
        self.addWidget('ComboBox', 'Method',
                      items=['SOS', 'Adaptive', 'Walsh'],
                      val=1)  # Default to Adaptive
        
        self.addWidget('DoubleSpinBox', 'Smoothing', val=5.0, min=0.0, max=20.0,
                      singlestep=0.5, decimals=1)
        
        self.addWidget('CheckBox', 'Auto-compute', val=True)
        
        self.addWidget('TextBox', 'Status', val='Ready')
    
    def validate(self):
        """Validate input data."""
        data = self.getData('data')
        
        try:
            if data.ndim < 3:
                raise ValueError(f"Data must be at least 3D [n_coils, ...], got {data.ndim}D")
            
            if not np.iscomplexobj(data):
                raise ValueError("Data must be complex-valued")
            
            self.setAttr('Status', val='Validation passed')
            return 0
            
        except ValueError as e:
            self.setAttr('Status', val=f'Error: {str(e)}')
            return 1
    
    def compute(self):
        """Estimate coil sensitivity maps."""
        logger = logging_config.get_logger('coil_sensitivity_node')
        
        # Get input data
        data = self.getData('data')
        
        # Get widget values
        method_idx = self.getVal('Method')
        smoothing = self.getVal('Smoothing')
        
        method_map = {0: 'sos', 1: 'adaptive', 2: 'walsh'}
        method = method_map[method_idx]
        
        logger.info(f"Estimating coil maps using {method} method")
        
        try:
            self.setAttr('Status', val=f'Estimating ({method})...')
            
            # Check if data is k-space or image domain
            # Simple heuristic: if data is mostly in corners, likely k-space
            is_kspace = self._is_kspace_data(data)
            
            if is_kspace:
                logger.info("Input appears to be k-space data")
                # Transform to image space
                if data.ndim == 3:  # 2D
                    coil_images = mri_math.ifft2c(data)
                else:  # 3D
                    coil_images = mri_math.ifft3c(data)
                
                # Estimate from k-space
                coil_maps = mri_math.estimate_coil_sensitivity(
                    data, method=method, smoothing=smoothing
                )
            else:
                logger.info("Input appears to be image data")
                coil_images = data
                
                # For image data, create k-space for estimation
                if data.ndim == 3:
                    kspace = mri_math.fft2c(data)
                else:
                    kspace = mri_math.fft3c(data)
                
                coil_maps = mri_math.estimate_coil_sensitivity(
                    kspace, method=method, smoothing=smoothing
                )
            
            # Combine using estimated maps
            combined = mri_math.sense_combine(coil_images, coil_maps)
            
            # Set outputs
            self.setData('coil_maps', coil_maps.astype(np.complex64))
            self.setData('combined_image', combined.astype(np.complex64))
            
            logger.info("Coil map estimation complete")
            self.setAttr('Status', val='Estimation complete!')
            
            return 0
            
        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            self.setAttr('Status', val=f'Error: {str(e)}')
            return 1
    
    def _is_kspace_data(self, data):
        """
        Heuristic to determine if data is k-space or image domain.
        
        K-space typically has high energy at the center and lower at edges.
        """
        # Get center region energy vs edge energy
        center_slice = tuple(slice(s//4, 3*s//4) for s in data.shape[1:])
        center_energy = np.sum(np.abs(data[(slice(None),) + center_slice])**2)
        
        edge_size = min(data.shape[1:]) // 8
        edge_slice = tuple(slice(0, edge_size) for _ in data.shape[1:])
        edge_energy = np.sum(np.abs(data[(slice(None),) + edge_slice])**2)
        
        # If center has much more energy than edge, likely k-space
        ratio = center_energy / (edge_energy + 1e-10)
        
        return ratio > 10.0  # Threshold for k-space detection


if __name__ == '__main__':
    if not HAS_GPI:
        print("GPI framework not available. This node requires GPI to run.")
