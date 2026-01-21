#ifndef MRIRECON_MRI_RECON_ARGS_HPP
#define MRIRECON_MRI_RECON_ARGS_HPP

#include "GPIArray/GPIArray.hpp" // For GPIArray::Array

namespace MRIRecon {

/**
 * @brief Struct to hold arguments specific to the DeblurFX operator.
 * @tparam T The real-valued type (float or double) for the numerical operations.
 */
template <typename T>
struct DeblurArgs {
  // fmap_residual: x, y, slice, coil
  GPIArray::Array<T> fmap_residual;
  // fx_mask: x, y, slice
  GPIArray::Array<uint64_t> fx_mask;
  // ksp_phase: x, y, z, te, water/fat, band
  GPIArray::Array<std::complex<T>> ksp_phase;
  // te_phase: x, y, z, te, band
  GPIArray::Array<std::complex<T>> te_phase;
  GPIArray::Array<std::complex<T>> kern_table;
  GPIArray::Array<T> kern_radius;
  GPIArray::Array<long> minindex;
  T freq_increment;

  // Default constructor
  DeblurArgs() = default;

  // Constructor with common parameters
  DeblurArgs(const GPIArray::Array<T>& fmap_res,
         const GPIArray::Array<uint64_t>& mask,
         const GPIArray::Array<std::complex<T>>& ksp_ph,
         const GPIArray::Array<std::complex<T>>& te_ph,
         const GPIArray::Array<std::complex<T>>& kern_tbl,
         const GPIArray::Array<T>& kern_rad,
         const GPIArray::Array<long>& min_idx,
         T freq_inc)
    : fmap_residual(fmap_res), fx_mask(mask), ksp_phase(ksp_ph),
      te_phase(te_ph), kern_table(kern_tbl), kern_radius(kern_rad),
      minindex(min_idx), freq_increment(freq_inc) {}
};

/**
 * @brief Struct to hold arguments specific to the Gridder operator.
 * @tparam T The real-valued type (float or double) for the numerical operations.
 */
template <typename T>
struct GridArgs {
    GPIArray::Array<T> coords;
    GPIArray::Array<T> sdc;
    GPIArray::Array<T> coords_cg; // Can be empty GPIArray
    GPIArray::Array<T> sdc_cg;    // Can be empty GPIArray
    uint64_t matrix_size;
    T oversample_factor;

    // Default constructor
    GridArgs() = default;

    // Constructor with common parameters
    GridArgs(const GPIArray::Array<T>& c,
             const GPIArray::Array<T>& s,
             uint64_t m_size,
             T over_fac)
        : coords(c), sdc(s), matrix_size(m_size), oversample_factor(over_fac) {}

    // Constructor with all parameters
    GridArgs(const GPIArray::Array<T>& c,
             const GPIArray::Array<T>& s,
             const GPIArray::Array<T>& c_cg,
             const GPIArray::Array<T>& s_cg,
             uint64_t m_size,
             T over_fac)
        : coords(c), sdc(s), coords_cg(c_cg), sdc_cg(s_cg),
          matrix_size(m_size), oversample_factor(over_fac) {}
};

/**
 * @brief Struct to hold parameters for the Conjugate Gradient optimization.
 * @tparam T The real-valued type (float or double) used in GPIArray.
 */
template <typename T>
struct OptimizationArgs {
  T lambda_reg = 0.01;                  // Tikhonov regularization parameter
  int max_iterations = 100;             // Maximum CG iterations
  T tolerance = 1e-3;                   // Convergence tolerance
  uint64_t wavelet_levels = 5;          // Wavelet decomposition levels
  bool use_d2_not_d4 = false;           // Use Daubechies-2 (Haar) wavelets
  enum class OptimizationMethod { CG, FISTA, ADMM };
  OptimizationMethod optimization_method = OptimizationMethod::CG; // Optimization algorithm to use

  // --- Parameters for FieldMapUpdater ---
  bool update_field_map = false; // Whether to update the field map
  GPIArray::Array<T> echo_times;          // Echo times in ms for field map calculation
  GPIArray::Array<T> fat_peaks;           // Fat peak frequencies (kHz) and relative amplitudes
  T field_map_damping_factor = 0.5;       // Damping factor for field map update stability

  OptimizationArgs() = default;

};

} // namespace MRIRecon

#endif // MRIRECON_MRI_RECON_ARGS_HPP
