# Copyright (c) 2014, Dignity Health
# 
#     The GPI core node library is licensed under
# either the BSD 3-clause or the LGPL v. 3.
# 
#     Under either license, the following additional term applies:
# 
#         NO CLINICAL USE.  THE SOFTWARE IS NOT INTENDED FOR COMMERCIAL
# PURPOSES AND SHOULD BE USED ONLY FOR NON-COMMERCIAL RESEARCH PURPOSES.  THE
# SOFTWARE MAY NOT IN ANY EVENT BE USED FOR ANY CLINICAL OR DIAGNOSTIC
# PURPOSES.  YOU ACKNOWLEDGE AND AGREE THAT THE SOFTWARE IS NOT INTENDED FOR
# USE IN ANY HIGH RISK OR STRICT LIABILITY ACTIVITY, INCLUDING BUT NOT LIMITED
# TO LIFE SUPPORT OR EMERGENCY MEDICAL OPERATIONS OR USES.  LICENSOR MAKES NO
# WARRANTY AND HAS NOR LIABILITY ARISING FROM ANY USE OF THE SOFTWARE IN ANY
# HIGH RISK OR STRICT LIABILITY ACTIVITIES.
# 
#     If you elect to license the GPI core node library under the LGPL the
# following applies:
# 
#         This file is part of the GPI core node library.
# 
#         The GPI core node library is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version. GPI core node library is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
# 
#         You should have received a copy of the GNU Lesser General Public
# License along with the GPI core node library. If not, see
# <http://www.gnu.org/licenses/>.


# Author: Guruprasad Krishnamoorthy, PhD
# Date: 2025July

import gpi
import numpy as np
from gpi import MRIData
import time
import pywt
import os

# Import modules once at module level
import spiral_prime.spiral_prime.CGMultiSlice as CGMultiSlice
import spiral_prime.spiral_prime.KernelBuilder as kernel
import spiral_prime.spiral_prime.DeblurFXArgs as DeblurFXArgs

from scipy.interpolate import interp1d

# OpenMP thread configuration
NTHREADS = 12  # Macbook Pro M2 Max
os.environ['OMP_NUM_THREADS'] = str(NTHREADS)

class ExternalNode(gpi.NodeAPI):
    """
    Spiral PRIME/SENSE reconstruction with water-fat separation, deblurring, and optional ADMM regularization.
    Supports both standard SENSE and PRIME (Phase Reconstruction In Multi-Echo) solvers with multi-peak fat
    modeling and optional 2D/3D wavelet denoising via ADMM optimization.

    INPUT:
        mri_data_in - MRI dataset containing k-space spiral data (coils, echoes, slices, arms, readout points)
        csm - coil sensitivity maps, 4D [coil, z, y, x]
        freq map (Hz) - B0 field map in Hz, 3D [z, y, x]
        band mask - (optional) per-band field map mask for localized deblurring
        mri_data_out - (optional) reference MRI data for PRIME solver

    OUTPUT:
        data_out - reconstructed water-fat images with shape [..., WF, (z), y, x]

    PARAMETERS:
        Solver - reconstruction algorithm: 'SENSE' (standard sensitivity encoding) or 'PRIME' (ADMM-based)
        Multipeak - fat model: 'No Fat' (water only), '1 peak' (single-peak fat), '7 peaks' (multi-peak fat model)
        B0 (Tesla) - scanner field strength (used to scale fat peak frequencies)
        side - kernel side extent in pixels (number of pixels on each side of main lobe)
        kern table phase inc (deg) - kernel table phase increment in degrees
        Max Iteration - maximum CG iterations for deblurring (0 = no CG, direct gridding only)
        Global Error Threshold (%) - convergence threshold for ADMM iterations (early stopping criterion)
    """

    def initUI(self):
        # Widgets
        self.addWidget('TextBox', 'Info:')
        self.addWidget('PushButton', 'Compute', toggle=True, val=False)
        self.addWidget('DoubleSpinBox', 'kern table phase inc (deg)', val=30, min=1., max=360)
        self.addWidget('SpinBox', 'side', val=8, min=0)
        self.addWidget('ExclusivePushButtons', 'Multipeak', buttons=['No Fat', '1 peak', '7 peaks'], val=2, visible=True)
        self.addWidget('DoubleSpinBox', 'B0 (Tesla)', val=3.0, min=0.1)
        self.addWidget('SpinBox', title='Max Iteration', min=0, val=4, max=100)
        self.addWidget('DoubleSpinBox', title='Global Error Threshold (%)', min=0.0, val=1e-3, decimals=9, max=1.0)
        self.addWidget('ExclusivePushButtons', 'Solver', buttons=['SENSE', 'PRIME'], val=0, visible=True)

        # IO Ports
        self.addInPort('mri_data_in', 'MRIDATA')
        self.addInPort('mri_data_out', 'MRIDATA', obligation=gpi.OPTIONAL)
        self.addInPort('csm', 'NPYarray', ndim=4)

        self.addInPort('freq map (Hz)', 'NPYarray',  ndim=3)
        self.addInPort('band mask', 'NPYarray', dtype=[np.int32], ndim=3, obligation=gpi.OPTIONAL)

        self.addOutPort(title='data_out', type='NPYarray')
        
    def validate(self):
        return 0

    def compute(self):

        if not self.getVal('Compute'):
            return 0

        # Input Data
        csm = self.getData('csm')
        mri_data_in = self.getData('mri_data_in')
        field_map = self.getData('freq map (Hz)')
        mask_band = self.getData('band mask')
        
        # Optional Input Data
        mri_data_out: MRIData = self.getData('mri_data_out')
        
        deblur_params = {
            "num_fat_peaks": self.getVal('Multipeak'),  # Number of fat peaks in the spectrum
            "kern_table_phase_inc": self.getVal('kern table phase inc (deg)') ,  # Kernel table phase increment in degrees
            "kernel_side": self.getVal('side') ,  # Kernel side length in pixels
            "b0_Tesla": self.getVal('B0 (Tesla)'),  # B0 field strength in Tesla
        }
        
        deblurfx_op = prepare_recon_args(mri_data_in, field_map, csm, mask_band, deblur_params, mri_data_out)
        
        solver = self.getVal('Solver')
        if solver == 1:  # PRIME
            final_image = admm_recon_numpy(
                deblurfx_op,
                max_cg_iter=self.getVal('Max Iteration'),
                eps=self.getVal('Global Error Threshold (%)'),
                fg_mask=deblurfx_op.get_bg_mask()
            )
        else:  # SENSE
            
            final_image = deblurfx_op.cg_solve(deblurfx_op.get_initial_image(), 1.0, self.getVal('Max Iteration'))
        
        final_image = np.flip(final_image, axis=(-1,-2, -4))  # Flip x, y and z-axis for correct orientation
        # Interpolate axis -4 (z-axis) by factor 1.25 using sinc interpolation
        old_size = final_image.shape[-4]
        new_size = int(np.round(old_size * 1.25))
        old_indices = np.arange(old_size)
        new_indices = np.linspace(0, old_size - 1, new_size)

        # Reshape to (slices, rest) for interpolation
        original_shape = final_image.shape
        final_image = final_image.reshape(old_size, -1)
        interpolator = interp1d(old_indices, final_image, axis=0, kind='cubic', fill_value='extrapolate')
        final_image = interpolator(new_indices)
        final_image = final_image.reshape((new_size,) + original_shape[1:])
        self.setData('data_out', final_image)

        return 0


def get_fat_peaks(num_peaks: int, b0Tesla: float) -> np.ndarray:
    """
    Returns fat peak amplitudes and frequencies (in kHz) scaled for the given B0 field.
    Args:
        num_peaks (int): 0 = no fat, 1 = single-peak, 2 = multi-peak (7 peaks)
        b0 (float): Field strength in mT
    Returns:
        np.ndarray: shape (7, 2), columns are [relative amplitude, frequency offset (kHz)]
    """
    b0 = b0Tesla * 1000.  # Convert Tesla to mT
    #########################
    # LOGIC FOR FAT PEAKS
    #########################
    peaks = np.zeros((7,2), dtype=np.float64)
    # weighted sum is: -382.172 Hz
    # fat relative to water
    #             %, kHz
    # NOTE THIS IS FOR 3T - below we scale for b0 != 3T
    if num_peaks == 2:
        peaks[0] = [ 4.2, 0.091]
        peaks[1] = [ 1.6, -.234]
        peaks[2] = [ 6.6, -.300]
        peaks[3] = [ 9.5, -.328]
        peaks[4] = [ 7.1, -.385]
        peaks[5] = [62.5, -.422]
        peaks[6] = [ 8.5, -.473]
    elif num_peaks == 1:       
        peaks[0] = [ 0, 0.091]
        peaks[1] = [ 0, -.234]
        peaks[2] = [ 0, -.300]
        peaks[3] = [ 0, -.328]
        peaks[4] = [ 0, -.385]
        peaks[5] = [100, -.422]
        peaks[6] = [ 0, -.473]
    elif num_peaks == 0:
        peaks[0] = [ 100, 0.0]
        peaks[1] = [ 0, -.234]
        peaks[2] = [ 0, -.300]
        peaks[3] = [ 0, -.328]
        peaks[4] = [ 0, -.385]
        peaks[5] = [ 0, -.422]
        peaks[6] = [ 0, -.473]
    if np.sum(peaks[:, 0]) != 100.0:
        print("WARNING: Relative isochramat contributions are not balanced!")
    peaks[:, 0] /= 100.0        
    # Scale for other fields besides 3T
    peaks[:, 1] *= -b0/3000.
    return peaks

#########################
# Compute band-wise field map mean and residuals
##########################
class BandwiseFieldMap:
    """
    Computes the mean and residuals of the field map for each frequency band.
    """

    def __init__(self, field_map_hz: np.ndarray, fx_mask: np.ndarray = None) -> None:
        self.field_map_hz = field_map_hz
        self.fx_mask = fx_mask
        self.fmap_mean = None
        self.fmap_residual = None
        self.num_bands = None
        self._compute()

    def _compute(self) -> None:
        field_map_hz = self.field_map_hz
        fx_mask = self.fx_mask

        if fx_mask is None:
            fx_mask = np.ones(field_map_hz.shape, dtype=np.int32)
            self.fx_mask = fx_mask

        num_slices = field_map_hz.shape[0]
        matrix_size = field_map_hz.shape[-1]
        num_bands = int(np.max(fx_mask))
        fmap_mean = np.zeros((num_bands, num_slices), dtype=np.float64)
        fmap_residual = np.zeros((num_bands, num_slices, matrix_size, matrix_size), dtype=np.float64)

        for band in range(num_bands):
            band_mask = (fx_mask == (band + 1)).astype(np.float64)
            denom = np.sum(band_mask, axis=(1, 2))
            denom[denom == 0] = 1.0  # Prevent division by zero
            fmap_mean[band] = np.sum(field_map_hz * band_mask, axis=(1, 2)) / denom
            fmap_residual[band] = (field_map_hz - fmap_mean[band, :, np.newaxis, np.newaxis]) * band_mask

        self.fmap_mean = fmap_mean
        self.fmap_residual = fmap_residual
        
def shift_spiral_trajectory(k_space_coords: np.ndarray, delay_us: float) -> np.ndarray:
    from scipy.interpolate import interp1d
    """
    Shifts a spiral MRI k-space trajectory by a specified time delay using
    quadratic interpolation.

    Args:
        k_space_coords (np.ndarray): A NumPy array of shape (N, 2) where N is the
                                     number of samples. The columns represent the
                                     k_x and k_y coordinates.
        delay_us (float): The time delay in microseconds by which to shift the
                          trajectory.

    Returns:
        np.ndarray: The shifted k-space coordinates of shape (N, 2).
    """

    num_samples = k_space_coords.shape[-2]
    n_echo = k_space_coords.shape[0]
    n_arms = k_space_coords.shape[1]
    dt_us = 2.0  # Time step between samples in microseconds

    # Create a time vector for the original trajectory
    time_points_us = np.arange(num_samples) * dt_us

    # Calculate the new time points after the delay
    shifted_time_points_us = time_points_us + delay_us
    
    for echo in range(n_echo):
        for arm in range(n_arms):
            shifted_kx = interp1d(time_points_us, k_space_coords[echo, arm, :, 0], kind='quadratic', fill_value="extrapolate")(shifted_time_points_us)
            shifted_ky = interp1d(time_points_us, k_space_coords[echo, arm, :, 1], kind='quadratic', fill_value="extrapolate")(shifted_time_points_us)

            # Combine the shifted kx and ky back into a single array
            shifted_k_space_coords = np.stack((shifted_kx, shifted_ky), axis=1)
            k_space_coords[echo, arm, :, :] = shifted_k_space_coords

    return k_space_coords

def prepare_recon_args(in_mri_data: MRIData, 
                       field_map: np.ndarray, 
                       csm: np.ndarray, 
                       band_mask: np.ndarray, 
                       deblur_params: dict,
                       out_mri_data: MRIData = None) -> object:
    
    num_fat_peaks = deblur_params["num_fat_peaks"]
    kern_table_phase_inc = deblur_params["kern_table_phase_inc"]
    kernel_side = deblur_params["kernel_side"]
    b0_Tesla = deblur_params["b0_Tesla"]

    data_dtype = in_mri_data.idata.dtype
    precession = 'float' if data_dtype == np.complex64 else 'double'

    deblurfx_class = CGMultiSlice.CGMultiSliceF if precession == 'float' else CGMultiSlice.CGMultiSliceD
    
    fat_peak_list = get_fat_peaks(num_fat_peaks, b0_Tesla)

    field_map_khz = 0.001 * field_map
    fx_map_obj = BandwiseFieldMap(field_map_khz, band_mask)

    kspace_data, grid_args, deblur_args = get_data_args(in_mri_data, fx_map_obj, fat_peak_list, precession, kern_table_phase_inc,
                    kernel_side, b0_Tesla)
    
    if out_mri_data:
        in_kspace_data, in_grid_args, in_deblur_args = get_data_args(out_mri_data, fx_map_obj, fat_peak_list, precession, kern_table_phase_inc,
                    kernel_side, b0_Tesla)
        
        deblurfx_op = deblurfx_class(kspace_data, csm, deblur_args, grid_args, in_kspace_data, in_deblur_args, in_grid_args)
    else:
        deblurfx_op = deblurfx_class(kspace_data, csm, deblur_args, grid_args)
    
    return deblurfx_op

def get_data_args(mri_data: MRIData, fx_map_obj: 'BandwiseFieldMap', fat_peak_list: np.ndarray, 
                  precision: str, kern_table_phase_inc: float,
                  kern_side: int, b0_Tesla: float) -> tuple:
        """
        Extracts k-space data, coordinates, and SDC from an MRIData object.
        Args:
            mri_data (MRIData): Input MRIData object.
        Returns:
            tuple: (kspace_data, coords, sdc)
        """
        kspace_data = mri_data.idata
        data_dtype = mri_data.idata.dtype
        precision = 'float' if data_dtype == np.complex64 else 'double' 
        grid_args = DeblurFXArgs.GridArgsF if precision == 'float' else DeblurFXArgs.GridArgsD
        deblur_args = DeblurFXArgs.DeblurArgsF if precision == 'float' else DeblurFXArgs.DeblurArgsD
        
        
        coords_in = mri_data.coordinates.coords
        sdc = mri_data.coordinates.sdc
        coords_cg = mri_data.coordinates.coords_cg
        sdc_cg = mri_data.coordinates.sdc_cg
        dwell_time = mri_data.data_params['spiral_gen_params']['dwell']
        gamma = mri_data.data_params['spiral_gen_params']['gamma']
        ech_times_us = mri_data.data_params['echo_times']

        if(mri_data.data_params['contrast'] == 'SE'):
            ech_times_us = np.array(ech_times_us) - ech_times_us[0]
        
        echo_times = ech_times_us * 1e-3  # Convert to milliseconds
        crds_arm = coords_in[0,0,:,:]
        
        in_out_dim = 'out'
        if (crds_arm[-1,0]*crds_arm[-1,0] + crds_arm[-1,1]*crds_arm[-1,1]) < (crds_arm[0,0]*crds_arm[0,0] + crds_arm[0,1]*crds_arm[0,1]):
            in_out_dim = 'in'

        if(in_out_dim == 'out'):
            coords_in = shift_spiral_trajectory(coords_in, delay_us=2)
        
        matrix_size = 1
        if mri_data.coordinates.tmap_in is not None and mri_data.coordinates.tmap_in.ndim == 2:
            matrix_size = mri_data.coordinates.tmap_in.shape[-1]
        elif mri_data.coordinates.tmap_out is not None and mri_data.coordinates.tmap_out.ndim == 2:
            matrix_size = mri_data.coordinates.tmap_out.shape[-2]
        num_coils, num_slices, num_tes, num_arms, num_pts_per_arm = kspace_data.shape
        
        echo_times = np.array(echo_times)
        if echo_times.shape[0] != num_tes:
            if echo_times.shape[0] > num_tes:
                echo_times = echo_times[:num_tes]
            else:
                raise ValueError(f"echo_times must have length {num_tes}, got {echo_times.shape[0]}")
        
        # Normalize kspace_data so that its maximum absolute value is 100
        max_abs = np.abs(kspace_data).max()
        if max_abs > 0:
            kspace_data = 10000 * kspace_data / max_abs
            
        # Prepare the data to match the expected input for the deblurfx_class
        kspace_data = np.reshape(kspace_data, (num_coils, num_slices, num_tes,  -1))
        coords = np.reshape(coords_in, (num_tes, -1, 2))
        sdc = np.reshape(sdc, (num_tes, -1))
        if coords_cg is not None:
            coords_cg = np.reshape(coords_cg, (num_tes, -1, 2))
            sdc_cg = np.reshape(sdc_cg, (num_tes, -1))
        
        # Create the grid_args instance
        grid_args_inst = grid_args()
        grid_args_inst.coords = coords
        grid_args_inst.sdc = sdc
        grid_args_inst.matrix_size = matrix_size
        grid_args_inst.oversample_factor = 2.0
        if coords_cg is not None and sdc_cg is not None:
            grid_args_inst.coords_cg = coords_cg
            grid_args_inst.sdc_cg = sdc_cg
            
        # Create the MultiBandKernels object containing kphase, tephase, mindex, kern, kernrad
        freq_increment = kern_table_phase_inc / (360.0 * dwell_time * float(num_pts_per_arm))

        b0_mT = 1000. * b0_Tesla  # Convert to mT
        side = kern_side
 
        if in_out_dim == 'in': 
            crds_arm = np.flip(crds_arm, axis=0)  # Flip the coordinates if spiral direction is inward
            echo_times *= -1.0
            
        ##################################################################################
        # Calculate some kernels and modulating arrays
        # w0(r) is field map
        # w_n is chem shift for fat peaks, weighted by a_n (the relative proportion, e.g. for 7-peak fat)
        # w_n can also be the average off-resonance for W and F
        # tau(k) is the time during measurement as a function of k-space location
        #  modulation over r (i.e. x,y} and k for the collected signal s is given by
        # s = Sum_n[ a_n*Exp{ i (w0(r)+w_n) * (te + tau(k)) } ]
        #     <------------------ offres & chem shift -------->
        #
        # Rewrite as:
        #
        # s = Exp[i w0(r) te] * Exp[i w0(r) tau(k)] * Sum_n[ a_n*Exp{ i (w_n) * (te + tau(k)) } ]
        #     |             |   |                 |   |                                         |
        #      \ tephase   /     \     kern      /     \              kphase                   /
        #      multiply in r       convolve in r               multiply in k-space
        ##################################################################################
        kern, kernrad, kphase, tephase, mindex,  = kernel.generate_kernels(
            crds_arm, fx_map_obj.fmap_residual, fx_map_obj.fmap_mean,
            fat_peak_list, echo_times,
            b0_mT, dwell_time, gamma,
            side, freq_increment
        )
        
        deblur_args_inst = deblur_args()
        deblur_args_inst.fmap_residual = fx_map_obj.fmap_residual
        deblur_args_inst.fx_mask = fx_map_obj.fx_mask
        
        if in_out_dim == 'in': 
            deblur_args_inst.ksp_phase = np.conj(kphase)
            deblur_args_inst.te_phase = np.conj(tephase)
            deblur_args_inst.kern_table = np.conj(kern)
        else:
            deblur_args_inst.ksp_phase = kphase
            deblur_args_inst.te_phase = tephase
            deblur_args_inst.kern_table = kern
            
        deblur_args_inst.kern_radius = kernrad
        deblur_args_inst.minindex = mindex
        deblur_args_inst.freq_increment = freq_increment

        return kspace_data, grid_args_inst, deblur_args_inst
    
def admm_recon_numpy(
    deblurfx_op: object, 
    rho_init: float = 1.0, 
    max_admm_iter: int = 30, 
    max_cg_iter: int = 4, 
    eps: float = 1e-8, 
    rho_max: float = 2.0, 
    gamma: float = 1.1, 
    beta: float = 1.0,
    verbose: bool = True,
    fg_mask: np.ndarray = None,
    use_2d_denoising: bool = False
) -> np.ndarray:
    """
    ADMM reconstruction using NumPy + DeblurFX operator + 3D wavelet prox.
    Expects complex arrays of shape (S, WF, H, W).
    """
    AHb = deblurfx_op.get_initial_image()  # numpy complex array
    x = AHb.copy()
    z = x.copy()
    u = np.zeros_like(x)

    rho = float(rho_init)
    
    t_start = time.time()
    for k in range(max_admm_iter):
        t0 = time.time()
        x_old = x.copy()
        rho_k = rho

        # 1) x-update: solve (A^H A + rho I) x = A^H b + rho (z - u)
        x = deblurfx_op.cg_solve(z - u, rho, max_cg_iter)
        
        if k == 0:
            # Build wavelet denoiser once (compute SURE τ on initial image)
            S = x.shape[0]  # number of slices
            if S == 1 or use_2d_denoising:
                wav_denoiser = WaveletDenoiser2DNP(x, mask=fg_mask)
            else:
                wav_denoiser = WaveletDenoiser3DNP(x, mask=fg_mask)
        
        # 1a) early stop on x-stability
        delta = np.mean(np.abs(x_old - x)**2)
        if k >= 2 and delta < eps:
            if verbose:
                print(f"Stopped at iter {k} (delta < {eps:g}). Total time {time.time()-t_start:.2f}s")
            break

        # 2) z-update: proximal (denoiser) on x + u
        prox_input = x + u
        tau_scale = beta * (rho_init / rho)  # update schedule
        # recompute_sure=True each iter is allowed but costly; flip to False if you want cached τ
        z, _ = wav_denoiser.denoise(prox_input, tau_scale=tau_scale, cycle_spins=2)

        # 3) u-update (scaled dual)
        u += x - z

        if verbose:
            print(
                f"iter {k:02d} | rho {rho_k:.3f} | tau_scale {tau_scale:.3f} | "
                f"Δ {delta:.3e} | "
                f"t {time.time()-t0:.2f}s"
            )

        # 4) rho schedule (with dual rescaling)
        rho_next = min(rho * gamma, rho_max)
        if rho_next != rho:
            u *= (rho / rho_next)
        rho = rho_next
    recon_image = x.copy()
    return recon_image

class WaveletDenoiser3DNP:
    """
    3D complex wavelet denoiser over (S,H,W) per WF channel.

    Constructor:
      initial_image: complex array (S, WF, H, W) — computes initial SURE τ.
      mask         : optional (S,H,W) boolean array.
    """

    RAYLEIGH_MEDIAN = 1.1774100225154747  # sqrt(2 ln 2)

    @staticmethod
    def _complex_soft(x: np.ndarray, tau: float, eps: float = 1e-12) -> np.ndarray:
        mag = np.abs(x)
        scale = np.maximum(0.0, 1.0 - tau / np.maximum(mag, eps))
        return x * scale

    @staticmethod
    def _sure_tau_rayleigh(absvals: np.ndarray, n_grid: int = 60) -> float:
        w = absvals.reshape(-1).astype(np.float32)
        N = w.size
        if N == 0:
            return 0.0
        sigma = max(float(np.median(w) / WaveletDenoiser3DNP.RAYLEIGH_MEDIAN), 1e-12)
        t_max = sigma * np.sqrt(2.0 * np.log(max(N, 2)))
        t_grid = np.linspace(0.0, 2.0 * t_max, num=int(n_grid), dtype=np.float32)
        w_sorted = np.sort(w)
        w2_sorted = w_sorted * w_sorted
        csum_w2 = np.cumsum(w2_sorted)
        k = np.searchsorted(w_sorted, t_grid, side="right")
        sum_le = np.where(k > 0, csum_w2[k - 1], 0.0)
        sig2 = sigma * sigma
        R = N * sig2 + sum_le + (N - k) * (t_grid * t_grid) - 2.0 * sig2 * k
        return float(t_grid[np.argmin(R)])

    def __init__(self,
                 initial_image: np.ndarray,
                 mask: np.ndarray = None,
                 *,
                 wavelet: str = "db4",
                 mode: str = "symmetric",
                 J: int = 3,
                 skip_coarsest: int = 0,
                 n_grid: int = 160) -> None:
        assert initial_image.ndim == 4 and np.iscomplexobj(initial_image)

        self.wavelet = wavelet
        self.mode = mode
        self.n_grid = int(n_grid)

        self.S, self.WF, self.H, self.W = initial_image.shape
        # Corrected: passing wavelet string directly to pywt.dwtn_max_level
        maxJ = pywt.dwtn_max_level((self.S, self.H, self.W), wavelet=self.wavelet)
        self.J = max(1, min(int(J), maxJ, 6))
        self.skip_coarsest = max(0, min(int(skip_coarsest), self.J))
        self._coarsest_start = self.J - self.skip_coarsest
        self.mask = mask.astype(bool) if mask is not None else None

        self.taus = self._compute_sure_taus(initial_image)

    def _downsample_mask_to(self, shape_3d: tuple) -> np.ndarray:
        if self.mask is None:
            return None
        ms = self.mask.shape
        cs = shape_3d
        idxS = np.minimum((np.arange(cs[0]) * ms[0] // cs[0]), ms[0]-1)
        idxH = np.minimum((np.arange(cs[1]) * ms[1] // cs[1]), ms[1]-1)
        idxW = np.minimum((np.arange(cs[2]) * ms[2] // cs[2]), ms[2]-1)
        return self.mask[np.ix_(idxS, idxH, idxW)]

    def _pool_vals_for_band(self, band: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return np.abs(band).ravel()
        m = self._downsample_mask_to(band.shape)
        return np.abs(band[m]).ravel()

    def _compute_sure_taus(self, volume: np.ndarray) -> list:
        taus_all = []
        for g in range(self.WF):
            vol_g = volume[:, g, :, :]
            coeffs = pywt.wavedecn(vol_g, wavelet=self.wavelet,
                                   mode=self.mode, level=self.J)
            taus_g = []
            for level_idx, detail_dict in enumerate(coeffs[1:], start=1):
                tau_dict = {}
                if level_idx >= self._coarsest_start + 1:
                    for bname in detail_dict.keys():
                        tau_dict[bname] = 0.0
                else:
                    for bname, band in detail_dict.items():
                        vals = self._pool_vals_for_band(band)
                        tau_dict[bname] = (0.0 if vals.size == 0
                                           else self._sure_tau_rayleigh(vals, n_grid=self.n_grid))
                taus_g.append(tau_dict)
            taus_all.append(taus_g)
        return taus_all

    def _single_denoise(self, volume: np.ndarray, tau_scale: float) -> np.ndarray:
        out = np.empty_like(volume)
        for g in range(self.WF):
            vol_g = volume[:, g, :, :]
            coeffs = pywt.wavedecn(vol_g, wavelet=self.wavelet,
                                   mode=self.mode, level=self.J)
            new_coeffs = [coeffs[0]]
            for detail_dict, tau_dict in zip(coeffs[1:], self.taus[g]):
                d_new = {}
                for bname, band in detail_dict.items():
                    tau = float(tau_dict.get(bname, 0.0)) * tau_scale
                    d_new[bname] = self._complex_soft(band, tau) if tau > 0 else band
                new_coeffs.append(d_new)
            rec = pywt.waverecn(new_coeffs, wavelet=self.wavelet, mode=self.mode)
            out[:, g, :, :] = rec[:self.S, :self.H, :self.W]
        return out

    def denoise(self,
                volume: np.ndarray,
                *,
                recompute_sure: bool = False,
                tau_scale: float = 1.0,
                cycle_spins: int = 1,
                reduce: str = "mean") -> tuple:
        assert volume.shape == (self.S, self.WF, self.H, self.W)
        if recompute_sure:
            self.taus = self._compute_sure_taus(volume)

        if cycle_spins <= 1:
            rec = self._single_denoise(volume, tau_scale)
            return rec, {"taus": self.taus, "cycle_spins": 1}

        # Cycle spinning shifts along (S,H,W)
        acc = []
        for dS in range(cycle_spins):
            for dH in range(cycle_spins):
                for dW in range(cycle_spins):
                    # Corrected: Shift only along S, H, W axes (0, 2, 3)
                    shifted = np.roll(volume, shift=(dS, dH, dW), axis=(0, 2, 3))
                    rec_shifted = self._single_denoise(shifted, tau_scale)
                    # Corrected: Unshift back by applying the negative shift to the same axes
                    unshifted = np.roll(rec_shifted, shift=(-dS, -dH, -dW), axis=(0, 2, 3))
                    acc.append(unshifted)
        stack = np.stack(acc, axis=0)
        if reduce == "mean":
            rec = stack.mean(axis=0)
        else:
            rec = np.median(stack, axis=0)
        return rec, {"taus": self.taus, "cycle_spins": cycle_spins, "reduce": reduce}
    
class WaveletDenoiser2DNP:
    """
    2D complex wavelet denoiser applied slice-by-slice over (H, W),
    with SURE τ computed per slice, WF channel by pooling coefficients within each slice.
    """

    RAYLEIGH_MEDIAN = 1.1774100225154747  # sqrt(2 ln 2)

    # Use skimage for robust downsampling if available, otherwise fall back to a manual method
    try:
        from skimage.transform import resize

        _SKIMAGE_AVAILABLE = True
    except ImportError:
        _SKIMAGE_AVAILABLE = False

    @staticmethod
    def _complex_soft(x: np.ndarray, tau: float, eps: float = 1e-12) -> np.ndarray:
        """Apply complex soft-thresholding to a complex array."""
        mag = np.abs(x)
        scale = np.maximum(0.0, 1.0 - tau / np.maximum(mag, eps))
        return x * scale

    @staticmethod
    def _sure_tau_rayleigh(absvals: np.ndarray, n_grid: int = 160) -> float:
        """
        Estimate SURE-optimal threshold for Rayleigh-distributed noise.
        Based on minimizing Stein's Unbiased Risk Estimate.
        """
        w = absvals.ravel().astype(np.float32)
        N = w.size
        if N == 0:
            return 0.0
        
        # Estimate Rayleigh noise scale parameter sigma
        sigma = max(float(np.median(w) / WaveletDenoiser2DNP.RAYLEIGH_MEDIAN), 1e-12)
        
        # Determine the search grid for the optimal threshold
        t_max = sigma * np.sqrt(2.0 * np.log(max(N, 2)))
        t_grid = np.linspace(0.0, 2.0 * t_max, num=int(n_grid), dtype=np.float32)
        
        # Use a pre-sorted array for faster calculation
        w_sorted = np.sort(w)
        w2_sorted = w_sorted * w_sorted
        csum_w2 = np.cumsum(w2_sorted)
        
        # Calculate SURE for each grid point
        k = np.searchsorted(w_sorted, t_grid, side="right")
        sum_le = np.where(k > 0, csum_w2[k - 1], 0.0)
        sig2 = sigma * sigma
        R = N * sig2 + sum_le + (N - k) * (t_grid * t_grid) - 2.0 * sig2 * k
        
        return float(t_grid[np.argmin(R)])

    def __init__(self,
                 initial_image: np.ndarray,
                 mask: np.ndarray = None,
                 *,
                 wavelet: str = "db4",
                 mode: str = "symmetric",
                 J: int = 3,
                 skip_coarsest: int = 0,
                 n_grid: int = 160) -> None:
        assert initial_image.ndim == 4 and np.iscomplexobj(initial_image), \
            "initial_image must be complex with shape (S, WF, H, W)"

        self.wavelet = wavelet
        self.mode = mode
        self.n_grid = int(n_grid)

        self.S, self.WF, self.H, self.W = initial_image.shape
        maxJ = pywt.dwtn_max_level((self.H, self.W), self.wavelet)
        self.J = max(1, min(int(J), maxJ, 6))
        self.skip_coarsest = max(0, min(int(skip_coarsest), self.J))
        self._coarsest_start = self.J - self.skip_coarsest

        self.mask = mask.astype(bool) if mask is not None else None

        # Precompute SURE τ on the provided initial image, per slice and WF
        self.taus = self._compute_sure_taus(initial_image)

    # ------------ helpers ------------
    def _downsample_mask2d(self, shape_2d: tuple) -> np.ndarray:
        """Nearest downsample mask per slice to (h, w). Returns (S, h, w) or None."""
        if self.mask is None:
            return None
        h, w = shape_2d
        
        # Correctly downsample the 3D mask using skimage or a manual method
        if WaveletDenoiser2DNP._SKIMAGE_AVAILABLE:
            return WaveletDenoiser2DNP.resize(self.mask, (self.S, h, w), order=0, preserve_range=True).astype(bool)
        else:
            # Fallback for when skimage is not available.
            H, W = self.H, self.W
            iH = np.minimum((np.arange(h) * H // h), H - 1)
            iW = np.minimum((np.arange(w) * W // w), W - 1)
            return self.mask[:, iH, :][:, :, iW]

    def _compute_sure_taus(self, volume: np.ndarray) -> list:
        """
        Compute τ per slice, WF × level × subband by pooling |coeffs| within each slice.
        Returns: list[S][WF] -> list(levels finest..coarsest) of dict{band: tau}.
        """
        S, WF, H, W = volume.shape
        taus_all = []

        for s in range(S):
            taus_s = []
            for g in range(WF):
                # Decompose this slice for this WF channel
                coeffs_g = pywt.wavedecn(volume[s, g], wavelet=self.wavelet, mode=self.mode, level=self.J)
                # Get the band names and shapes from this decomposition
                band_names_per_level = [list(d.keys()) for d in coeffs_g[1:]]

                # Precompute masks downsampled per level for this slice
                ds_masks_per_level = []
                for lvl_idx, detail_dict in enumerate(coeffs_g[1:], start=1):
                    first_band = next(iter(detail_dict.values()))
                    ds_masks_per_level.append(self._downsample_mask2d(first_band.shape))

                taus_g = []
                for lvl_idx, band_names in enumerate(band_names_per_level, start=1):
                    tau_dict = {}
                    if lvl_idx >= self._coarsest_start + 1:
                        for bname in band_names:
                            tau_dict[bname] = 0.0
                    else:
                        for bname in band_names:
                            band = coeffs_g[lvl_idx][bname]
                            m_ds = ds_masks_per_level[lvl_idx - 1]
                            if self.mask is None:
                                vals = np.abs(band).ravel()
                            elif m_ds is not None:
                                vals = np.abs(band[m_ds[s]]).ravel()
                            tau_dict[bname] = 0.0 if vals.size == 0 else self._sure_tau_rayleigh(vals, n_grid=self.n_grid)
                    taus_g.append(tau_dict)
                taus_s.append(taus_g)
            taus_all.append(taus_s)
        return taus_all

    def _single_slice_denoise2d(self, img2d: np.ndarray, taus_g: list) -> np.ndarray:
        """
        Denoise a single 2D complex image using precomputed taus_g (list over levels).
        """
        coeffs = pywt.wavedecn(img2d, wavelet=self.wavelet, mode=self.mode, level=self.J)
        new_coeffs = [coeffs[0]]
        for detail_dict, tau_dict in zip(coeffs[1:], taus_g):
            d_new = {}
            for bname, band in detail_dict.items():
                tau = float(tau_dict.get(bname, 0.0))
                d_new[bname] = self._complex_soft(band, tau) if tau > 0.0 else band
            new_coeffs.append(d_new)
        rec = pywt.waverecn(new_coeffs, wavelet=self.wavelet, mode=self.mode)
        return rec[:self.H, :self.W]

    def _single_denoise_volume(self, volume: np.ndarray, tau_scale: float) -> np.ndarray:
        """
        Apply 2D denoising to each slice using τ scaled by tau_scale, per WF.
        """
        out = np.empty_like(volume)
        for s in range(self.S):
            for g in range(self.WF):
                # Scale taus for this slice and WF
                taus_scaled = [{k: float(v) * tau_scale for k, v in lvl.items()} for lvl in self.taus[s][g]]
                out[s, g] = self._single_slice_denoise2d(volume[s, g], taus_scaled)
        return out

    # ------------ public API ------------
    def denoise(self,
                volume: np.ndarray,
                *,
                recompute_sure: bool = False,
                tau_scale: float = 1.0,
                cycle_spins: int = 1,
                reduce: str = "mean") -> tuple:
        """
        volume        : complex array (S, WF, H, W)
        recompute_sure: if True, recompute SURE τ on this volume (per slice) before denoising
        tau_scale     : multiply all τ by this factor
        cycle_spins   : integer K => shifts dy,dx ∈ {0..K-1}; aggregation over unshifted outputs
        reduce        : 'mean' or 'median'
        """
        assert volume.shape == (self.S, self.WF, self.H, self.W) and np.iscomplexobj(volume)

        if recompute_sure:
            self.taus = self._compute_sure_taus(volume)

        if cycle_spins <= 1:
            rec = self._single_denoise_volume(volume, tau_scale)
            return rec, {"taus": self.taus, "cycle_spins": 1}

        # Cycle spinning over H,W only
        acc = []
        for dy in range(cycle_spins):
            for dx in range(cycle_spins):
                # Correct np.roll usage
                shifted = np.roll(volume, shift=(dy, dx), axis=(-2, -1))
                rec_shifted = self._single_denoise_volume(shifted, tau_scale)
                # Correct unshift
                unshifted = np.roll(rec_shifted, shift=(-dy, -dx), axis=(-2, -1))
                acc.append(unshifted)
        
        stack = np.stack(acc, axis=0)
        if reduce == "mean":
            rec = stack.mean(axis=0)
        else:
            rec = np.median(stack, axis=0)
            
        return rec, {"taus": self.taus, "cycle_spins": cycle_spins, "reduce": reduce}