#ifndef MRIRECON_FISTA_SOLVER_HPP
#define MRIRECON_FISTA_SOLVER_HPP

#include "GPIArray/GPIArray.hpp"
#include "DeblurFXArgs.hpp"
#include "DeblurFX.hpp"
#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <algorithm> // Required for std::sort

using namespace GPIArray;

namespace MRIRecon {

template <typename T>
class FISTASolver {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "FISTASolver class only supports double or float data.");

private:
    using ComplexT = std::complex<T>;

    const DeblurFX<T>& _deblur_op;
    const DeblurFX<T>& _deblur_op2; // Instance of the DeblurFX operator (store by value)
    const OptimizationArgs<T> _opt_args;
    bool in_out_recon = false; // Flag for input/output reconstruction
    const Wavelet<ComplexT> _wavelet_op;
    // A fixed step size is a practical starting point. May need tuning.
    const T _step_size = 0.1;

    void soft_threshold(Array<ComplexT>& wavelet_coeffs, T threshold) const {
        if (threshold <= 0) return;
        for (uint64_t j = 0; j < wavelet_coeffs.size(1); ++j) {
            for (uint64_t i = 0; i < wavelet_coeffs.size(0); ++i) {
                T magnitude = std::abs(wavelet_coeffs(i, j));
                if (magnitude > 1e-9) { 
                    T new_magnitude = std::max(static_cast<T>(0.0), magnitude - threshold);
                    wavelet_coeffs(i, j) *= (new_magnitude / magnitude);
                } else {
                    wavelet_coeffs(i, j) = 0;
                }
            }
        }
    }

    // Donoho, D. L., & Johnstone, I. M. (1995). Adapting to unknown smoothness via wavelet shrinkage. 
    // Journal of the American Statistical Association, 90(432), 1200-1224.
    T estimate_noise_sigma(const Array<ComplexT>& wavelet_coeffs) const {
        uint64_t w_rows = _wavelet_op.get_wavelet_size1();
        uint64_t w_cols = _wavelet_op.get_wavelet_size2();
        // Extract the HH1 band (high-high frequency band)
        // This is the band that contains the highest frequency details.
        // It is typically located in the bottom-right corner of the wavelet coefficients.
        auto hh1_band = wavelet_coeffs.slice(S(w_rows / 2, w_rows), S(w_cols / 2, w_cols));

        // std::string folder_name = "/Users/gkrishnamoo3/Documents/Data/Exam9898_05Jun2025_vvol_composite/";
        // std::string file_name = folder_name + "wav_coefff.npy";
        // if (!Numpy::file_exists(file_name)) {
        //     Numpy::write_npy(wavelet_coeffs, file_name);
        // }

        // file_name = folder_name + "hh1_band.npy";
        // if (!Numpy::file_exists(file_name)) {
        //     Numpy::write_npy(hh1_band, file_name);
        // }

        if (hh1_band.size() == 0) return 1e-9;

        std::vector<T> hh1_abs_vals;
        hh1_abs_vals.reserve(hh1_band.size());
        const ComplexT* data_ptr = hh1_band.get_data();
        for (uint64_t i = 0; i < hh1_band.size(); ++i) {
            hh1_abs_vals.push_back(std::abs(data_ptr[i]));
        }
        
        // Filter out coefficients less than or equal to 0
        std::vector<T> positive_vals;
        for (const auto& val : hh1_abs_vals) {
            if (val > 0) {
            positive_vals.push_back(val);
            }
        }
        if (positive_vals.empty()) return 1e-9;

        std::sort(positive_vals.begin(), positive_vals.end());
        T median = positive_vals[positive_vals.size() / 2];
        // The constant 0.6745 is used to estimate the standard deviation from the median absolute deviation (MAD).
        // This is a common practice in robust statistics.
        return median / 0.6745;
    }

    Array<ComplexT> apply_proximal_operator(const Array<ComplexT>& img) const {
        // **IMPROVEMENT: Check master lambda. If 0, skip all denoising.**
        if (_opt_args.lambda_reg <= 0) {
            return img; // Return the image untouched
        }

        Array<ComplexT> denoised_img = img.empty_like();

        for (uint64_t wf = 0; wf < img.size(2); ++wf) {
            auto channel_slice = img.slice(S::all(), S::all(), S(wf));
            auto wavelet_coeffs = _wavelet_op.forward_transform(channel_slice);
            
            T sigma = estimate_noise_sigma(wavelet_coeffs);
            // Universal threshold scales the master lambda by the estimated noise.
            T lambda_now = _opt_args.lambda_reg * sigma * std::sqrt(2.0 * std::log(static_cast<T>(channel_slice.size())));
            soft_threshold(wavelet_coeffs, lambda_now);

            auto denoised_channel = _wavelet_op.inverse_transform(wavelet_coeffs);
            denoised_img.slice(S::all(), S::all(), S(wf)) = denoised_channel;
        }
        return denoised_img;
    }

public:
    FISTASolver(const DeblurFX<T>& deblur_op_in,
                const OptimizationArgs<T>& opt_args_in)
        : _deblur_op(deblur_op_in),
          _deblur_op2(deblur_op_in),
          _opt_args(opt_args_in),
          in_out_recon(false),
          _wavelet_op(deblur_op_in.get_matrix_size(),
                      deblur_op_in.get_matrix_size(),
                      opt_args_in.wavelet_levels,
                      opt_args_in.use_d2_not_d4)
    {
    }

    FISTASolver(const DeblurFX<T>& deblur_op_in,
                const DeblurFX<T>& deblur_op_out,
                const OptimizationArgs<T>& opt_args_in)
        : _deblur_op(deblur_op_in),
          _deblur_op2(deblur_op_out),
          _opt_args(opt_args_in),
          in_out_recon(true),
          _wavelet_op(deblur_op_in.get_matrix_size(),
                      deblur_op_in.get_matrix_size(),
                      opt_args_in.wavelet_levels,
                      opt_args_in.use_d2_not_d4)
    {
    }

    Array<ComplexT> solve(const Array<ComplexT>& b,
    const Array<ComplexT>& b1 = Array<ComplexT>()) const {
        Array<ComplexT> x_k = _deblur_op.adjoint_op(b);
        
        // If in_out_recon is true, we also compute x_k for the second b
        if (in_out_recon) {
            x_k += _deblur_op2.adjoint_op(b1);
        }
        Array<ComplexT> x_prev = x_k.copy();
        Array<ComplexT> y_k = x_k.copy();
        T t_k = 1.0, t_prev = 1.0;

        for (int iter = 0; iter < _opt_args.max_iterations; ++iter) {
            // --- Step 1: Gradient descent ---

            Array<ComplexT> grad = _deblur_op.adjoint_op(_deblur_op.forward_op(y_k) - b);

            // If in_out_recon is true, we also compute the gradient for the second operator
            if(in_out_recon) {
                grad += _deblur_op2.adjoint_op(_deblur_op2.forward_op(y_k) - b1);
            }
            Array<ComplexT> z_k = y_k - _step_size * grad;

            // --- Step 2: Proximal step with adaptive thresholding ---
            x_k = apply_proximal_operator(z_k);

            // --- Step 3: Nesterov acceleration update ---
            t_k = (1.0 + std::sqrt(1.0 + 4.0 * t_prev * t_prev)) / 2.0;
            y_k = x_k + ((t_prev - 1.0) / t_k) * (x_k - x_prev);
            
            x_prev = x_k.copy();
            t_prev = t_k;
            
            T relative_change = l2norm(y_k - x_prev) / (l2norm(x_prev) + 1e-9);
            //std::cout << "Iteration " << iter << ", relative change: " << relative_change << std::endl;
            if (iter > 1 && relative_change < _opt_args.tolerance) {
                
                break;
            }
        }

        return x_k;
    }
};

} // namespace MRIRecon

#endif // MRIRECON_FISTA_SOLVER_HPP