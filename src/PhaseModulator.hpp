#ifndef PHASEMOD_HPP
#define PHASEMOD_HPP

#include "GPIArray/GPIArray.hpp" // Ensure Array.hpp is included
#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>

// Ensure M_PI is defined, if not already by <cmath> on all platforms
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace GPIArray;

namespace MRIRecon {
template <typename T>
class PhaseModulator {
    // Compile-time check for supported real types for complex numbers
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "PhaseModulator class only supports std::complex<double> or std::complex<float> data.");

private:
    // Member variables to store pre-computed kernels and phases
    Array<std::complex<T>> water_kphase_;
    Array<std::complex<T>> fat_kphase_;
    Array<std::complex<T>> water_kphase_conj_; // Conjugate of kphase
    Array<std::complex<T>> fat_kphase_conj_; // Conjugate of kphase

    uint64_t matrix_size_ = 256; // Matrix size for k-space coordinates

    std::shared_ptr<FFTW::FFTPlanManager<T>> fft2d_plan;

    //========================================================================
    // NEW: Phase Map Pre-Shifting Function - Uses FFTW::fftshift
    //========================================================================
    /**
     * @brief Performs FFT-shift (center alignment) on all 2D planes of the stored k-space phase maps.
     * This uses the existing FFTW::fftshift<T>() utility.
     * @param ksp_phase_in The input array containing UN-SHIFTED K-SPACE phase factors.
     */
    void pre_shift_phase_maps(const Array<std::complex<T>>& ksp_phase_in)
    {
        // 1. Setup dimensions
        matrix_size_ = ksp_phase_in.size(0);
        uint64_t num_te = ksp_phase_in.size(2);
        uint64_t num_bands = ksp_phase_in.size(4); // Assuming 5D: [mtx, mtx, te, 2, band]

        int ndims = ksp_phase_in.ndim(); 
        if (ndims == 4) {
            num_bands = 1;
        }

        // 2. Create owning copies of the K-space maps (4D array [mtx, mtx, te, band])
        // These are the large arrays we will modify in place.
        water_kphase_ = ksp_phase_in.slice(S::all(), S::all(), S::all(), S(0), S::all()).copy();
        fat_kphase_ = ksp_phase_in.slice(S::all(), S::all(), S::all(), S(1), S::all()).copy();

        // 3. Iterate over all 2D planes and perform the contiguous shift operation.
        for (uint64_t b = 0; b < num_bands; ++b) {
            for (uint64_t t = 0; t < num_te; ++t) {
                
                // --- Slice to get the current target plane views (2D) ---
                // These slices are the non-contiguous destinations in the member arrays.
                Array<std::complex<T>> water_target_view = water_kphase_.slice(S::all(), S::all(), S(t), S(b));
                Array<std::complex<T>> fat_target_view = fat_kphase_.slice(S::all(), S::all(), S(t), S(b));

                // --- 4. Water Plane: Copy IN, Shift, Assign OUT ---
                
                // Copy IN: Create a *deep copy* of the target view. This resulting array is GUARANTEED contiguous.
                Array<std::complex<T>> temp_plane_w = water_target_view.copy(); 
                
                // Perform the Shift IN-PLACE on the contiguous temp_plane_w
                FFTW::fftshift<T>(temp_plane_w); 
                
                // Assign OUT: Assign the shifted contiguous array back to the non-contiguous target view.
                // The Array::operator= should handle the element-wise copy/assignment correctly.
                water_target_view = temp_plane_w;

                // --- 5. Fat Plane: Copy IN, Shift, Assign OUT ---
                
                // Copy IN: Create a *deep copy* for guaranteed contiguity.
                Array<std::complex<T>> temp_plane_f = fat_target_view.copy(); 
                
                // Perform the Shift IN-PLACE on the contiguous temp_plane_f
                FFTW::fftshift<T>(temp_plane_f); 
                
                // Assign OUT: Assign the shifted contiguous array back to the non-contiguous target view.
                fat_target_view = temp_plane_f;
            }
        }

        // 6. Pre-compute conjugates of the *shifted* k-phases.
        water_kphase_conj_ = conj(water_kphase_);
        fat_kphase_conj_ = conj(fat_kphase_);
    }


    //========================================================================
    // Modulate phase in k-space
    //
    // @brief Private helper function to modulate phase for water and fat signals.
    // This function performs in-place FFTs on copies of the input arrays to
    // avoid modifying the original data.
    //========================================================================
    void modulate_phase_impl(Array<std::complex<T>>& water_in,
                             Array<std::complex<T>>& fat_in,
                             const Array<std::complex<T>>& w_kphase,
                             const Array<std::complex<T>>& f_kphase,
                             Array<std::complex<T>>& out) const
    {
        fft2d_plan->execute_forward(water_in, false);
        fft2d_plan->execute_forward(fat_in, false);

        // Modulate with respective k-phases and sum water and fat.
        // This manual loop is highly optimized for performance as it avoids
        // creating temporary Array objects for the element-wise multiplication.
        for (uint64_t j = 0; j < water_in.size(1); ++j) {
            for (uint64_t i = 0; i < water_in.size(0); ++i) {
                out(i, j) = water_in(i, j) * w_kphase(i, j) + fat_in(i, j) * f_kphase(i, j);
            }
        }

        // Go back to image space using in-place FFT.
        fft2d_plan->execute_backward(out, false);
    }

    
    //========================================================================
    // Demodulate phase in k-space
    //
    // @brief Private helper function to demodulate a complex image signal.
    // It performs in-place FFTs on a copy of the input to ensure data integrity.
    //========================================================================
    void demodulate_phase_impl(Array<std::complex<T>> &in,
                               const Array<std::complex<T>> &w_kphase,
                               const Array<std::complex<T>> &f_kphase,
                               Array<std::complex<T>> &w_out,
                               Array<std::complex<T>> &f_out) const
    {
        // Go to k-space using in-place FFT.
        fft2d_plan->execute_forward(in, false);

        // Demodulate the water and fat components by element-wise multiplication with k-phases.
        // The manual loop avoids temporary Array allocations, improving performance.
        for (uint64_t j = 0; j < in.size(1); ++j) {
            for (uint64_t i = 0; i < in.size(0); ++i) {
                w_out(i, j) = in(i, j) * w_kphase(i, j);
                f_out(i, j) = in(i, j) * f_kphase(i, j);
            }
        }

        // Go back to image space using in-place FFTs on the output arrays.
        fft2d_plan->execute_backward(w_out, false);
        fft2d_plan->execute_backward(f_out, false);
    }

public:
    // Default constructor is enabled for flexibility.
    PhaseModulator() = default;

    /**
     * @brief Constructor that pre-computes phase conjugates and initializes the FFTW plan.
     * @param ksp_phase A 5D array of dimensions [mtx_size, mtx_size, num_te, 2, num_bands]
     */
    PhaseModulator(const Array<std::complex<T>>& ksp_phase)
    {
        // Validate input dimensions.
        if (ksp_phase.ndim() < 4 || ksp_phase.ndim() > 5)
            THROW_RUNTIME_ERROR("ksp_phase must be a 4D or 5D array with dimensions [mtx_size, mtx_size, num_te, 2, num_bands]");

        // Set the matrix size from the first dimension of the input.
        matrix_size_ = ksp_phase.size(0);

        pre_shift_phase_maps(ksp_phase);

        fft2d_plan = std::make_shared<FFTW::FFTPlanManager<T>>(std::vector<uint64_t>{matrix_size_, matrix_size_}, FFTW_MEASURE);
    }

    PhaseModulator(const Array<std::complex<T>>& ksp_phase, 
        std::shared_ptr<FFTW::FFTPlanManager<T>> fft2d_plan_in):
        fft2d_plan(fft2d_plan_in)
    {
        // Validate input dimensions.
        if (ksp_phase.ndim() < 4 || ksp_phase.ndim() > 5)
            THROW_RUNTIME_ERROR("ksp_phase must be a 4D or 5D array with dimensions [mtx_size, mtx_size, num_te, 2, num_bands]");

        // Set the matrix size from the first dimension of the input.
        matrix_size_ = ksp_phase.size(0);

        pre_shift_phase_maps(ksp_phase);
    }


    

    /**
     * @brief Public interface for phase demodulation.
     * @param in_image The input image to demodulate (const reference).
     * @param water_out The output array for the demodulated water component (modified in-place).
     * @param fat_out The output array for the demodulated fat component (modified in-place).
     * @param te_index The echo time index.
     * @param band_index The band index.
     */
    void demodulate_phase(Array<std::complex<T>> &in_image,
                          Array<std::complex<T>> &water_out,
                          Array<std::complex<T>> &fat_out,
                          const uint16_t te_index = 0,
                          const uint16_t band_index = 0) const
    {
        // Create lightweight views (slices) of the pre-computed conjugate k-phases.
        const auto w_kpa_conj_sliced = water_kphase_conj_.slice(S::all(), S::all(), S(te_index), S(band_index));
        const auto f_kpa_conj_sliced = fat_kphase_conj_.slice(S::all(), S::all(), S(te_index), S(band_index));
        
        // Delegate to the private implementation function.
        demodulate_phase_impl(in_image, w_kpa_conj_sliced, f_kpa_conj_sliced, water_out, fat_out);
    }

    /**
     * @brief Public interface for phase modulation.
     * @param water_in The input water image.
     * @param fat_in The input fat image.
     * @param out The output array for the modulated image (modified in-place).
     * @param te_index The echo time index.
     * @param band_index The band index.
     */
    Array<std::complex<T>> modulate_phase(Array<std::complex<T>> &water_in,
                        Array<std::complex<T>> &fat_in,
                        const uint16_t te_index = 0,
                        const uint16_t band_index = 0) const
    {
        // Slice the pre-computed k-phases for the specified TE and band.
        const auto w_kpa_sliced = water_kphase_.slice(S::all(), S::all(), S(te_index), S(band_index));
        const auto f_kpa_sliced = fat_kphase_.slice(S::all(), S::all(), S(te_index), S(band_index));

        Array<std::complex<T>> out(matrix_size_, matrix_size_);

        // Delegate to the private implementation function.
        modulate_phase_impl(water_in, fat_in, w_kpa_sliced, f_kpa_sliced, out);

        return std::move(out);
    }
};

} // namespace MRIRecon

#endif // PHASEMOD_HPP