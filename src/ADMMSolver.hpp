#ifndef MRIRECON_ADMM_SOLVER_HPP
#define MRIRECON_ADMM_SOLVER_HPP

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

template<typename T>
struct ADMMAux {
    T rho = 1.0; // ADMM penalty parameter
    int cg_iterations = 3; // Iterations for the conjugate gradient solver
};

template <typename T>
class ADMMSolver {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "ADMMSolver class only supports double or float data.");

private:
    using ComplexT = std::complex<T>;

    const DeblurFX<T>& _deblur_op;
    const DeblurFX<T>& _deblur_op2; // Instance of the DeblurFX operator
    const OptimizationArgs<T> _opt_args;
    const ADMMAux<T> _admm_aux;
    bool in_out_recon = false; // Flag for input/output reconstruction
    const Wavelet<ComplexT> _wavelet_op;

    mutable Array<ComplexT> _temp_denoised;
    mutable Array<ComplexT> _temp_x_prev;
    mutable Array<ComplexT> _temp_wavelet_coeffs;

    void soft_threshold(Array<ComplexT>& wavelet_coeffs, T threshold) const {
        if (threshold <= 0) return;
        
        ComplexT* data = wavelet_coeffs.get_data();
        uint64_t total_size = wavelet_coeffs.size();
        
        for (uint64_t i = 0; i < total_size; ++i) {
            T magnitude = std::abs(data[i]);
            if (magnitude > threshold) {
                data[i] *= (magnitude - threshold) / magnitude;
            } else {
                data[i] = ComplexT(0);
            }
        }
    }

    T estimate_noise_sigma(const Array<ComplexT>& wavelet_coeffs) const {
        uint64_t w_rows = _wavelet_op.get_wavelet_size1();
        uint64_t w_cols = _wavelet_op.get_wavelet_size2();
        auto hh1_band = wavelet_coeffs.slice(S(w_rows / 2, w_rows), S(w_cols / 2, w_cols));

        if (hh1_band.size() == 0) return 1e-9;

        std::vector<T> hh1_abs_vals;
        hh1_abs_vals.reserve(hh1_band.size());
        const ComplexT* data_ptr = hh1_band.get_data();
        for (uint64_t i = 0; i < hh1_band.size(); ++i) {
            hh1_abs_vals.push_back(std::abs(data_ptr[i]));
        }

        std::vector<T> positive_vals;
        for (const auto& val : hh1_abs_vals) {
            if (val > 0) {
                positive_vals.push_back(val);
            }
        }
        if (positive_vals.empty()) return 1e-9;

        std::sort(positive_vals.begin(), positive_vals.end());
        T median = positive_vals[positive_vals.size() / 2];
        return median / 0.6745;
    }

    Array<ComplexT> apply_proximal_operator(const Array<ComplexT>& img, T rho) const {
        if (_opt_args.lambda_reg <= 0) {
            return img;
        }

        Array<ComplexT> denoised_img = img.empty_like();

        for (uint64_t wf = 0; wf < img.size(2); ++wf) {
            auto channel_slice = img.slice(S::all(), S::all(), S(wf));
            auto wavelet_coeffs = _wavelet_op.forward_transform(channel_slice);

            T sigma = estimate_noise_sigma(wavelet_coeffs);
            T lambda_now = _opt_args.lambda_reg * sigma * std::sqrt(2.0 * std::log(static_cast<T>(channel_slice.size())));
            
            soft_threshold(wavelet_coeffs, lambda_now / rho);

            auto denoised_channel = _wavelet_op.inverse_transform(wavelet_coeffs);
            denoised_img.slice(S::all(), S::all(), S(wf)) = denoised_channel;
        }
        return denoised_img;
    }

    // Conjugate gradient to solve the x-update: (A'A + rho*I)x = b_cg
    Array<ComplexT> conjugate_gradient_solve(const Array<ComplexT>& b_cg, T rho) const {
        Array<ComplexT> x = b_cg.empty_like(); // Initial guess for x
        x.fill(0.0);
        Array<ComplexT> r = b_cg; // r = b_cg - (A'A + rho*I)x; since x is 0, r = b_cg
        Array<ComplexT> p = r.copy();
        T rs_old = std::real(sum(conj(r) * r));

        for (int i = 0; i < _admm_aux.cg_iterations; ++i) {
            Array<ComplexT> Ap = _deblur_op.adjoint_op(_deblur_op.forward_op(p));
            if(in_out_recon) {
                Ap += _deblur_op2.adjoint_op(_deblur_op2.forward_op(p));
            }
            Ap += rho * p;  // In-place addition instead of creating Ap_rho
            T alpha = rs_old / std::real(sum(conj(p) * Ap));
            
            x += static_cast<ComplexT>(alpha) * p;
            r -= static_cast<ComplexT>(alpha) * Ap;
            
            T rs_new = std::real(sum(conj(r) * r));
            if (std::sqrt(rs_new) < 1e-10) {
                break;
            }
            
            p = r + static_cast<ComplexT>(rs_new / rs_old) * p;
            rs_old = rs_new;
        }
        return x;
    }


public:
    ADMMSolver(const DeblurFX<T>& deblur_op_in,
                const OptimizationArgs<T>& opt_args_in,
                const ADMMAux<T>& admm_aux_in = ADMMAux<T>())
        : _deblur_op(deblur_op_in),
          _deblur_op2(deblur_op_in), // Placeholder
          _opt_args(opt_args_in),
          _admm_aux(admm_aux_in),
          in_out_recon(false),
          _wavelet_op(deblur_op_in.get_matrix_size(),
                      deblur_op_in.get_matrix_size(),
                      opt_args_in.wavelet_levels,
                      opt_args_in.use_d2_not_d4)
    {
        uint64_t matrix_size = deblur_op_in.get_matrix_size();
        _temp_denoised = Array<ComplexT>(matrix_size, matrix_size, 2);
        _temp_x_prev = Array<ComplexT>(matrix_size, matrix_size, 2);
    }

    ADMMSolver(const DeblurFX<T>& deblur_op_in,
                const DeblurFX<T>& deblur_op_out,
                const OptimizationArgs<T>& opt_args_in,
                const ADMMAux<T>& admm_aux_in = ADMMAux<T>())
        : _deblur_op(deblur_op_in),
          _deblur_op2(deblur_op_out),
          _opt_args(opt_args_in),
          _admm_aux(admm_aux_in),
          in_out_recon(true),
          _wavelet_op(deblur_op_in.get_matrix_size(),
                      deblur_op_in.get_matrix_size(),
                      opt_args_in.wavelet_levels,
                      opt_args_in.use_d2_not_d4)
    {
        uint64_t matrix_size = deblur_op_in.get_matrix_size();
        _temp_denoised = Array<ComplexT>(matrix_size, matrix_size, 2);
        _temp_x_prev = Array<ComplexT>(matrix_size, matrix_size, 2);
    }

    Array<ComplexT> solve(const Array<ComplexT>& b,
    const Array<ComplexT>& b1 = Array<ComplexT>()) const {
        
        // Initialization
        Array<ComplexT> x_k = _deblur_op.adjoint_op(b);
        if (in_out_recon) {
            x_k += _deblur_op2.adjoint_op(b1);
        }
        Array<ComplexT> z_k = x_k.copy();
        Array<ComplexT> u_k = x_k.empty_like();
        u_k.fill(0.0);
        
        Array<ComplexT> atb = _deblur_op.adjoint_op(b);
        if(in_out_recon){
            atb += _deblur_op2.adjoint_op(b1);
        }

        for (int iter = 0; iter < _opt_args.max_iterations; ++iter) {
            Array<ComplexT> x_prev = x_k.copy();

            // --- Step 1: x-update (solve (A'A + rho*I)x = A'b + rho*(z-u)) ---
            Array<ComplexT> cg_rhs = atb + _admm_aux.rho * (z_k - u_k);
            x_k = conjugate_gradient_solve(cg_rhs, _admm_aux.rho);
            
            // --- Step 2: z-update (proximal operator) ---
            z_k = apply_proximal_operator(x_k + u_k, _admm_aux.rho);

            // --- Step 3: u-update ---
            u_k += (x_k - z_k);
            
            T relative_change = l2norm(x_k - x_prev) / (l2norm(x_prev) + 1e-9);
            //std::cout << "Iteration " << iter << ", relative change: " << relative_change << std::endl;
            if (iter > 1 && relative_change < _opt_args.tolerance) {
                break;
            }
        }

        return x_k;
    }
};

} // namespace MRIRecon

#endif // MRIRECON_ADMM_SOLVER_HPP