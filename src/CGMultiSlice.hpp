#ifndef MRIRECON_CG_TIKHONOV_SOLVER_HPP
#define MRIRECON_CG_TIKHONOV_SOLVER_HPP

#include <iostream>
#include <complex>
#include <cmath> // For std::sqrt

#include "GPIArray/GPIArray.hpp"
#include "DeblurFX.hpp"
#include "DeblurFXArgs.hpp"
#include <omp.h>
#include <chrono>

using namespace GPIArray;

namespace MRIRecon {

template <typename T>
class CGMultiSlice {
    // Compile-time check for supported real types for complex numbers
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "CGMultiSlice class only supports double or float data.");

private:
    DeblurArgs<T> deb_args;
    GridArgs<T> grid_args;
    DeblurArgs<T> deb_args2;
    GridArgs<T> grid_args2;
    Array<std::complex<T>> csm; // Coil sensitivity map
    Array<std::complex<T>> ksp; // K-space data, samplesXarms, shots, slices, coils
    Array<std::complex<T>> ksp_out;
    bool in_out_recon = false; // Flag for input/output reconstruction


    std::vector<DeblurFX<T>> deblur_op;
    std::vector<DeblurFX<T>> deblur_op2;

    uint64_t num_slices; // Number of slices in the multi-slice reconstruction
    uint64_t matrix_size; // Size of the matrix for each slice
    T norm_factor = 1.0; // Normalization factor for k-space data
    Array<std::complex<T>> Ahb;
    bool is_scale_computed = false; // Flag to check if scale has been computed+
    Array<std::complex<T>> kmask; // K-space mask to suppress noise outside of sampled circular k-space
    Array<bool> bg_mask; // Background mask to estimate noise level

    Array<std::complex<T>> adjoint_op() {

        // Create an initial image array filled with zeros
        Array<std::complex<T>> output_image(matrix_size, matrix_size, 2, num_slices);
        output_image.fill(std::complex<T>(0));

        Array<std::complex<T>> zero_slice(matrix_size, matrix_size, 2);
        zero_slice.fill(std::complex<T>(0));

        #pragma omp parallel for
        for (uint64_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
                auto curr_mask = bg_mask.slice(S::all(), S::all(), S(slice_idx));
                if (sum(curr_mask) == 0) {
                    output_image.slice(S::all(), S::all(), S::all(), S(slice_idx)) = zero_slice;
                    continue;
                }
                auto ksp_slice = ksp.slice(S::all(), S::all(), S(slice_idx), S::all());
                auto initial_image = deblur_op[slice_idx].adjoint_op(ksp_slice);
                if (in_out_recon) {
                    auto ksp_out_slice = ksp_out.slice(S::all(), S::all(), S(slice_idx), S::all());
                    initial_image += deblur_op2[slice_idx].adjoint_op(ksp_out_slice);
                }
                output_image.slice(S::all(), S::all(), S::all(), S(slice_idx)) = initial_image;
        }
        return output_image;
    }

    void apply_kmask(Array<std::complex<T>>& in) const {
        // Apply k-space mask to suppress noise outside of sampled circular k-space
            for (uint64_t i = 0; i < in.size(2); ++i) {
                auto wf_slice = in.slice(S::all(), S::all(), S(i)).copy();
                FFTW::fft2(wf_slice, wf_slice, FFTW_FORWARD);
                for (uint64_t j = 0; j < wf_slice.size(1); ++j) {
                    for (uint64_t k = 0; k < wf_slice.size(0); ++k) {
                        wf_slice(j, k) *= kmask(j, k);
                    }
                }
                FFTW::fft2(wf_slice, wf_slice, FFTW_BACKWARD);
                in.slice(S::all(), S::all(), S(i)) = wf_slice;
            }
    }

        // Compute k-space mask to suppress noise outside of sampled circular k-space
    void compute_kmask(const Array<T>& x_coords, const Array<T>& y_coords) {
        Array<T> ktrace_all(x_coords.size(0));
        for (uint64_t i = 0; i < x_coords.size(0); ++i) {
        T x_val = x_coords(i, 0);
        T y_val = y_coords(i, 0);
        ktrace_all(i) = std::sqrt(x_val * x_val + y_val * y_val);
        }
        T max_ktrace = max(ktrace_all);

        T kmask_width = std::ceil(max_ktrace * static_cast<T>(matrix_size));
        kmask.resize(matrix_size, matrix_size);
        for (uint64_t i = 0; i < matrix_size; ++i) {
        for (uint64_t j = 0; j < matrix_size; ++j) {
            T norm = std::sqrt(
            std::pow(static_cast<T>(i) - static_cast<T>(matrix_size) / 2, 2) +
            std::pow(static_cast<T>(j) - static_cast<T>(matrix_size) / 2, 2)
            );
            T val = 0.5 + (1.0 / M_PI) * std::atan(100.0 * (1.0 - norm / kmask_width));
            kmask(i, j) = std::complex<T>(val, 0);
        }
        }
    };

public:
    CGMultiSlice(const Array<std::complex<T>>& ksp_in,
                 const Array<std::complex<T>>& csm_in,
                 const DeblurArgs<T>& deb_args1,
                 const GridArgs<T>& grid_args1,
                 const Array<std::complex<T>>& ksp_out_ = Array<std::complex<T>>(),
                 const DeblurArgs<T>& deb_args_out = DeblurArgs<T>(),
                 const GridArgs<T>& grid_args_out = GridArgs<T>())
        : ksp(ksp_in),
          csm(csm_in),
          deb_args(deb_args1),
          grid_args(grid_args1),
          matrix_size(grid_args1.matrix_size),
          num_slices(csm_in.size(2)),
          deb_args2(deb_args_out),
          grid_args2(grid_args_out),
          ksp_out(ksp_out_),
          in_out_recon(!ksp_out.is_empty())
    {
        //std::cout << "Initializing CGMultiSlice with " << num_slices << " slices." << std::endl;
        
        ksp /= (0.0001 * max(abs(ksp))); // Normalize input k-space
        if(in_out_recon) {
            ksp_out /= (0.0001 * max(abs(ksp_out))); // Normalize output k-space
        }
        
        deblur_op.resize(num_slices);
        auto x_coords = grid_args.coords.slice(S(0), S::all(), S::all());
        auto y_coords = grid_args.coords.slice(S(1), S::all(), S::all());
        Gridder<T> grid_op(x_coords, y_coords, grid_args.sdc, matrix_size, grid_args.oversample_factor);

        Gridder<T> grid_cg_op;
        if(grid_args.coords_cg.size() > 0 && grid_args.sdc_cg.size() > 0) {
            auto x_coords_cg = grid_args.coords_cg.slice(S(0), S::all(), S::all());
            auto y_coords_cg = grid_args.coords_cg.slice(S(1), S::all(), S::all());
            grid_cg_op = Gridder<T>(
                x_coords_cg, y_coords_cg, grid_args.sdc_cg,
                matrix_size, grid_args.oversample_factor
            );
        } else {
            // If no CG coordinates are provided, use the same grid_op for CG
            grid_cg_op = std::move(grid_op);
        }

        Gridder<T> grid_op_out;
        Gridder<T> grid_cg_op_out;
        if (in_out_recon) {
            deblur_op2.resize(num_slices);
            auto x_coords_out = grid_args2.coords.slice(S(0), S::all(), S::all());
            auto y_coords_out = grid_args2.coords.slice(S(1), S::all(), S::all());
            grid_op_out = Gridder<T>(x_coords_out, y_coords_out, grid_args2.sdc, matrix_size, grid_args2.oversample_factor);

            
            if(grid_args2.coords_cg.size() > 0 && grid_args2.sdc_cg.size() > 0) {
                auto x_coords_cg_out = grid_args2.coords_cg.slice(S(0), S::all(), S::all());
                auto y_coords_cg_out = grid_args2.coords_cg.slice(S(1), S::all(), S::all());
                grid_cg_op_out = Gridder<T>(
                    x_coords_cg_out, y_coords_cg_out, grid_args2.sdc_cg,
                    matrix_size, grid_args2.oversample_factor
                );
            } else {
                // If no CG coordinates are provided, use the same grid_op for CG
                grid_cg_op_out = std::move(grid_op_out);
            }

        }

        auto fft2d_plan = std::make_shared<FFTW::FFTPlanManager<T>>(std::vector<uint64_t>{matrix_size, matrix_size}, FFTW_MEASURE);
        // Initialize deblur operators for each slice using OpenMP
        #pragma omp parallel for
        for (uint64_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
                auto csm_slice = csm.slice(S::all(), S::all(), S(slice_idx), S::all());
                auto fmap_residual = deb_args.fmap_residual.slice(S::all(), S::all(), S(slice_idx), S::all());
                auto fx_mask = deb_args.fx_mask.slice(S::all(), S::all(), S(slice_idx));
                auto ksp_phase = deb_args.ksp_phase.slice(S::all(), S::all(), S(slice_idx), S::all(), S::all(), S::all());
                auto te_phase = deb_args.te_phase.slice(S::all(), S::all(), S(slice_idx), S::all(), S::all());

                PhaseModulator<T> phase_mod_op(ksp_phase, fft2d_plan);
                SpatVarConv<T> spat_conv_op(deb_args.kern_table, deb_args.kern_radius, deb_args.minindex, te_phase, fmap_residual, fx_mask, deb_args.freq_increment);

                deblur_op[slice_idx] = DeblurFX<T>(grid_op, grid_cg_op, spat_conv_op, phase_mod_op, csm_slice);

                if (in_out_recon) {
                    // If output k-space is provided, initialize the deblur operator for output
                    auto fmap_residual_out = deb_args2.fmap_residual.slice(S::all(), S::all(), S(slice_idx), S::all());
                    auto fx_mask_out = deb_args2.fx_mask.slice(S::all(), S::all(), S(slice_idx));
                    auto ksp_phase_out = deb_args2.ksp_phase.slice(S::all(), S::all(), S(slice_idx), S::all(), S::all(), S::all());
                    auto te_phase_out = deb_args2.te_phase.slice(S::all(), S::all(), S(slice_idx), S::all(), S::all());
                    PhaseModulator<T> phase_mod_op_out(ksp_phase_out, fft2d_plan);
                    SpatVarConv<T> spat_conv_op_out(deb_args2.kern_table, deb_args2.kern_radius, deb_args2.minindex, te_phase_out, fmap_residual_out, fx_mask_out, deb_args2.freq_increment);
                    deblur_op2[slice_idx] = DeblurFX<T>(grid_op_out, grid_cg_op_out, spat_conv_op_out, phase_mod_op_out, csm_slice);
                }
        }

        compute_kmask(x_coords, y_coords);

        // Initialize background mask based on fx_mask
        bg_mask.resize(matrix_size, matrix_size, num_slices);
        #pragma omp parallel for collapse(3)
        for (uint64_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
                for (uint64_t i = 0; i < matrix_size; ++i) {
                for (uint64_t j = 0; j < matrix_size; ++j) {
                    bg_mask(i, j, slice_idx) = deb_args.fx_mask(i, j, slice_idx) ? true : false;
                }
                }
        }

        // Compute normalization factor for k-space data
        Ahb.resize(matrix_size, matrix_size, 2, num_slices);
        Ahb = adjoint_op();

        ksp = Array<std::complex<T>>(); // Release ksp memory to save space
        ksp_out = Array<std::complex<T>>(); // Release ksp_out memory to save space
    }

    Array<std::complex<T>> cg_solve(const Array<std::complex<T>>& curr_image,
                                 T rho, uint64_t max_iterations)
    {
        auto x = curr_image.empty_like();
        Array<std::complex<T>> zero_slice(matrix_size, matrix_size, 2);
        zero_slice.fill(std::complex<T>(0));

        #pragma omp parallel for collapse(1)
        for (uint64_t slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
            
            auto curr_mask = bg_mask.slice(S::all(), S::all(), S(slice_idx));
            auto fg_pixel_count = sum(curr_mask);
            if(fg_pixel_count == 0) {
            // If no foreground pixels, set output to zero and skip
            x.slice(S::all(), S::all(), S::all(), S(slice_idx)) = zero_slice;
            continue;
            }

            auto curr_slice = curr_image.slice(S::all(), S::all(), S::all(), S(slice_idx));

            // Initial guess x = zeros
            Array<std::complex<T>> x_slice = curr_slice.empty_like();
            x_slice.fill(std::complex<T>(0)); // Start with zero guess

            // Compute initial residual r = rhs - (F^H F + rho I) x
            // Since x is initialized to zero, r = rhs
            auto Ahb_slice = Ahb.slice(S::all(), S::all(), S::all(), S(slice_idx));
            auto r = Ahb_slice + rho * curr_slice;

            auto p = r.copy(); // Initial search direction, same as r
            T rsold = std::real(dot(r, r)); // r^T r

            // Pre-allocate workspace arrays once per slice
            Array<std::complex<T>> Ap = curr_slice.empty_like();

            for (uint64_t i = 0; i < max_iterations; ++i) {
            // Ap = (F^H F + rho I) p
            Ap = deblur_op[slice_idx].adjoint_op(deblur_op[slice_idx].forward_op(p));
            if(in_out_recon) {
                // If output k-space is provided, use the output deblur operator
                Ap += deblur_op2[slice_idx].adjoint_op(deblur_op2[slice_idx].forward_op(p));
            }

            Ap += (rho * p);   
            // alpha = (r_k^T r_k) / (p_k^T A p_k)
            T alpha = rsold / std::real(dot(p, Ap));

            x_slice = x_slice + alpha * p;
            r = r - alpha * Ap;

            // // only 4 ietrations for now so skip convergence check
            T r_norm_sq = std::real(dot(r, r)); // For convergence check: ||r_k+1||^2
            if (std::sqrt(r_norm_sq) < 1e-3) {
                std::cout << "Converged at iteration " << i << " for slice " << slice_idx << std::endl;
                break;
            }

            // beta = (r_k+1^T r_k+1) / (r_k^T r_k)
            T rsnew = r_norm_sq; // r_new^T r_new
            p = r + (rsnew / rsold) * p; // Update search direction
            rsold = rsnew; // Update rsold for next iteration
            }
            apply_kmask(x_slice); // Apply k-space mask to the final image
            x.slice(S::all(), S::all(), S::all(), S(slice_idx)) = x_slice;
        }

        
        return x;
    }

    Array<std::complex<T>> get_initial_image() const {
        return Ahb;
    }

    const Array<bool>& get_bg_mask() const {
        return bg_mask;
    }

};

}
#endif // MRIRECON_CG_TIKHONOV_SOLVER_HPP
