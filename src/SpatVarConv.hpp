#ifndef SVCONV_HPP
#define SVCONV_HPP

#include "GPIArray/GPIArray.hpp"
#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono> // For timing, similar to original files

// Ensure M_PI is defined, if not already by <cmath> on all platforms
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


using namespace GPIArray;
namespace MRIRecon {
template <typename T>
class SpatVarConv {
    // Compile-time check for supported real types for complex numbers
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "SpatVarConv class only supports std::complex<double> or std::complex<float> data.");
private:

    Array<std::complex<T>> kern_table_;
    Array<T> kern_radius_;
    Array<long> min_index_;
    Array<std::complex<T>> te_phase_;
    Array<std::complex<T>> te_phase_conj_;
    Array<T> fmap_residual_;
    Array<uint64_t> fx_mask_;
    Array<uint64_t> i_pts_;
    Array<uint64_t> j_pts_;
    Array<uint64_t> ij_ctr_;
    T freq_inc_;
    uint64_t matrix_size_;
    Array<std::complex<T>> fx_mask_band; // Frequency mask for bands (0, 1)

    uint64_t num_te_ = 0; // Number of echo times
    uint64_t num_bands_ = 0; // Number of bands
    uint64_t num_coils_ = 0; // Number of coils

    //========================================================================
    // do_deblur_kern to do deblur according to the locally blurring kernel
    //========================================================================
    void do_deblur_kern(const Array<std::complex<T> > &in_w,
                   const Array<std::complex<T> > &in_f,
                   const Array<T> &map,
                   const Array<uint64_t> &ipts,
                   const Array<uint64_t> &jpts,
                   const uint64_t ij_ctr,
                   Array<std::complex<T> > &out,
                   const Array<std::complex<T> > &kern,
                   const Array <T> &kernrad,
                   const Array <long> &mindex,
                   const Array<std::complex<T> > &tephase,
                   const T freqinc,
                const uint64_t bandindex) const 
    {
        //auto begin = std::chrono::high_resolution_clock::now();
        
        uint64_t l2 = kern.size(0)-1; // kernel radius // Xi Peng, add -1
        
        // big_x arrays allow convolution without limit checks
        Array<std::complex<T> > big_w(in_w.size(0)+2*l2,in_w.size(1)+2*l2);
        Array<std::complex<T> > big_f(in_w.size(0)+2*l2,in_w.size(1)+2*l2);
        
        // transfer data to big_x arrays
        big_w.fill(std::complex<T>(0.));
        big_f.fill(std::complex<T>(0.));
        for (uint64_t j=0; j< in_w.size(1); j++) { // Loop over rows (y-axis)
            uint64_t jj = j+l2;
            for (uint64_t i=0; i< in_w.size(0); i++) { // Loop over columns (x-axis) - innermost
                uint64_t ii = i+l2;
                big_w(ii,jj) = in_w(i,j);
                big_f(ii,jj) = in_f(i,j);
            }
        } // i, j loop

        for(uint64_t i = 0; i < fx_mask_.size(0); i++) {
            for(uint64_t j = 0; j < fx_mask_.size(1); j++) {
                if(fx_mask_(i, j) == 0) {
                    out(i, j, 0) = in_w(i, j);
                    out(i, j, 1) = in_f(i, j);
                } else if(fx_mask_(i, j) == bandindex + 1) {
                    std::complex<T> te_phase_val = tephase(i,j); // Store te_phase value
                    if (std::abs(map(i, j)) > static_cast<T>(1e-6)) {
                        T frq_norm = (map(i,j)/freqinc);
                        uint64_t m0 = (std::floor)(frq_norm) - mindex(0); // kernel selection based on freq
                        uint64_t m1 = m0+1;
                        std::complex<T> wm1_val = static_cast<std::complex<T>>(frq_norm - (std::floor)(frq_norm));
                        std::complex<T> wm0_val = static_cast<std::complex<T>>(1.0) - wm1_val;

                        uint64_t ii = i+l2;
                        uint64_t jj = j+l2;
                        uint64_t lwid = (uint64_t) kernrad(m0); // Store kernrad value

                        // 1X symmetry at origin
                        std::complex<T> kval = std::conj(wm0_val*kern(0,0,m0) + wm1_val*kern(0,0,m1));
                        std::complex<T> tmpw = big_w(ii,jj)*kval;
                        std::complex<T> tmpf = big_f(ii,jj)*kval;          
                        
                        for (uint64_t la = 1; la <= lwid; la++) {
                            // 4X symmetry on axes
                            kval = std::conj(wm0_val*kern(la,0,m0) + wm1_val*kern(la,0,m1));
                            tmpw += kval*(big_w(ii+la,jj) +
                                big_w(ii-la,jj) +
                                big_w(ii,jj+la) +
                                big_w(ii,jj-la));
                            tmpf += kval*(big_f(ii+la,jj) +
                                big_f(ii-la,jj) +
                                big_f(ii,jj+la) +
                                big_f(ii,jj-la));

                            // 4X symmetry on diagonals
                            kval = std::conj(wm0_val*kern(la,la,m0) + wm1_val*kern(la,la,m1));
                            tmpw += kval*(big_w(ii+la,jj+la) +
                                big_w(ii-la,jj+la) +
                                big_w(ii+la,jj-la) +
                                big_w(ii-la,jj-la));
                            tmpf += kval*(big_f(ii+la,jj+la) +
                                big_f(ii-la,jj+la) +
                                big_f(ii+la,jj-la) +
                                big_f(ii-la,jj-la));

                            for (uint64_t lb=1; lb < la; lb++) {
                                // 8X symmetry everywhere else
                                kval = std::conj(wm0_val*kern(la,lb,m0) + wm1_val*kern(la,lb,m1));
                                tmpw += kval*(big_w(ii+la,jj+lb) +
                                        big_w(ii-la,jj+lb) +
                                        big_w(ii+la,jj-lb) +
                                        big_w(ii-la,jj-lb) +
                                        big_w(ii+lb,jj+la) +
                                        big_w(ii-lb,jj+la) +
                                        big_w(ii+lb,jj-la) +
                                        big_w(ii-lb,jj-la));
                                tmpf += kval*(big_f(ii+la,jj+lb) +
                                        big_f(ii-la,jj+lb) +
                                        big_f(ii+la,jj-lb) +
                                        big_f(ii-la,jj-lb) +
                                        big_f(ii+lb,jj+la) +
                                        big_f(ii-lb,jj+la) +
                                        big_f(ii+lb,jj-la) +
                                        big_f(ii-lb,jj-la));
                            }
                        } // la loop
                        
                        out(i,j,0) += tmpw*te_phase_val;
                        out(i,j,1) += tmpf*te_phase_val;
                    }
                    else {
                        out(i,j,0) += in_w(i,j)*te_phase_val;
                        out(i,j,1) += in_f(i,j)*te_phase_val;
                    } // map(i,j) != 0
                }
            }
        }   
    } // do_deblur_kernel

    //========================================================================
    // do_blur_kern: Applies spatially-varying blur kernel to input data
    //========================================================================
    void do_blur_kern(
        const Array<std::complex<T>> &in,
        const Array<T> &map,
        const Array<uint64_t> &ipts,
        const Array<uint64_t> &jpts,
        uint64_t ij_ctr,
        Array<std::complex<T>> &w_out,
        Array<std::complex<T>> &f_out,
        const Array<std::complex<T>> &kern,
        const Array<T> &kernrad,
        const Array<long> &mindex,
        const Array<std::complex<T>> &tephase,
        const T freqinc,
        const uint64_t bandindex
    ) const 
    {
        uint64_t l2 = kern.size(0) - 1; // kernel radius

        // big_x arrays allow convolution without limit checks
        Array<std::complex<T>> big_w(in.size(0) + 2 * l2, in.size(1) + 2 * l2);
        Array<std::complex<T>> big_f(in.size(0) + 2 * l2, in.size(1) + 2 * l2);

        // BLUR
        big_w.fill(std::complex<T>(0.));
        big_f.fill(std::complex<T>(0.));

        for (uint64_t i = 0; i < fx_mask_.size(0); ++i) {
            for (uint64_t j = 0; j < fx_mask_.size(1); ++j) {
                if (fx_mask_(i, j) == 0) { // Background pixels
                    big_w(i + l2, j + l2) = in(i, j, 0);
                    big_f(i + l2, j + l2) = in(i, j, 1);
                } 
                else if (fx_mask_(i, j) == bandindex+1)
                {
                    std::complex<T> te_phase_val = tephase(i, j); // Store te_phase value
                    std::complex<T> tmpw = in(i, j, 0) * te_phase_val;
                    std::complex<T> tmpf = in(i, j, 1) * te_phase_val;
                    uint64_t ii = i + l2;
                    uint64_t jj = j + l2;

                    if (std::abs(map(i, j)) > static_cast<T>(1e-6)) {
                        T frq_norm = (map(i, j) / freqinc);
                        uint64_t m0 = static_cast<uint64_t>((std::floor)(frq_norm) - mindex(0));
                        uint64_t m1 = m0 + 1;
                        T wm1 = frq_norm - (std::floor)(frq_norm);
                        T wm0 = 1. - wm1;
                        uint64_t lwid = static_cast<uint64_t>(kernrad(m0)); // Store kernrad value

                        // 1X symmetry at origin
                        std::complex<T> kval = wm0 * kern(0, 0, m0) + wm1 * kern(0, 0, m1);
                        big_w(ii, jj) += tmpw * kval;
                        big_f(ii, jj) += tmpf * kval;

                        for (uint64_t la = 1; la <= lwid; la++) {
                            // 4X symmetry on axes
                            kval = wm0 * kern(la, 0, m0) + wm1 * kern(la, 0, m1);
                            std::complex<T> tmpw_k = tmpw * kval;
                            std::complex<T> tmpf_k = tmpf * kval;

                            big_w(ii + la, jj) += tmpw_k;
                            big_w(ii - la, jj) += tmpw_k;
                            big_w(ii, jj + la) += tmpw_k;
                            big_w(ii, jj - la) += tmpw_k;

                            big_f(ii + la, jj) += tmpf_k;
                            big_f(ii - la, jj) += tmpf_k;
                            big_f(ii, jj + la) += tmpf_k;
                            big_f(ii, jj - la) += tmpf_k;

                            // 4X symmetry on diagonals
                            kval = wm0 * kern(la, la, m0) + wm1 * kern(la, la, m1);
                            tmpw_k = tmpw * kval;
                            tmpf_k = tmpf * kval;

                            big_w(ii + la, jj + la) += tmpw_k;
                            big_w(ii - la, jj + la) += tmpw_k;
                            big_w(ii + la, jj - la) += tmpw_k;
                            big_w(ii - la, jj - la) += tmpw_k;

                            big_f(ii + la, jj + la) += tmpf_k;
                            big_f(ii - la, jj + la) += tmpf_k;
                            big_f(ii + la, jj - la) += tmpf_k;
                            big_f(ii - la, jj - la) += tmpf_k;

                            for (uint64_t lb = 1; lb < la; lb++) {
                                // 8X symmetry everywhere else
                                kval = wm0 * kern(la, lb, m0) + wm1 * kern(la, lb, m1);
                                tmpw_k = tmpw * kval;
                                tmpf_k = tmpf * kval;

                                big_w(ii + la, jj + lb) += tmpw_k;
                                big_w(ii - la, jj + lb) += tmpw_k;
                                big_w(ii + la, jj - lb) += tmpw_k;
                                big_w(ii - la, jj - lb) += tmpw_k;
                                big_w(ii + lb, jj + la) += tmpw_k;
                                big_w(ii - lb, jj + la) += tmpw_k;
                                big_w(ii + lb, jj - la) += tmpw_k;
                                big_w(ii - lb, jj - la) += tmpw_k;

                                big_f(ii + la, jj + lb) += tmpf_k;
                                big_f(ii - la, jj + lb) += tmpf_k;
                                big_f(ii + la, jj - lb) += tmpf_k;
                                big_f(ii - la, jj - lb) += tmpf_k;
                                big_f(ii + lb, jj + la) += tmpf_k;
                                big_f(ii - lb, jj + la) += tmpf_k;
                                big_f(ii + lb, jj - la) += tmpf_k;
                                big_f(ii - lb, jj - la) += tmpf_k;
                            }
                        }
                    } else {
                        big_w(ii, jj) = tmpw;
                        big_f(ii, jj) = tmpf;
                    }
                }
            }
        }

        for (uint64_t j = 0; j < in.size(1); j++) { // Loop over rows (y-axis)
            for (uint64_t i = 0; i < in.size(0); i++) { // Loop over columns (x-axis) - innermost
                uint64_t ii = i + l2;
                uint64_t jj = j + l2;
                w_out(i, j) = big_w(ii, jj);
                f_out(i, j) = big_f(ii, jj);
            }
        }
    }

public:

    SpatVarConv() = default; // Default constructor

    SpatVarConv(const Array<std::complex<T>> &kern_table, 
                const Array<T> &kern_radius,
                const Array<long> &min_index,
                const Array<std::complex<T>> &te_phase,
                const Array<T> &fmap_residual,
                const Array<uint64_t> &fx_mask,
                const T freq_inc)
        : kern_table_(kern_table),
          kern_radius_(kern_radius),
          min_index_(min_index),
          te_phase_(te_phase),
          te_phase_conj_(conj(te_phase)),
          fmap_residual_(fmap_residual),
          fx_mask_(fx_mask),
          freq_inc_(freq_inc)
    {
        // Initialization of i_pts_, j_pts_, ij_ctr_ should go here if needed
        matrix_size_ = fx_mask_.size(0); // Assuming fx_mask_ is 2D and has tsize matrix_size x matrix_size

        num_bands_ = max(fx_mask_); // Number of frequency bands
        num_te_ = te_phase_.size(2); // Number of echo times

        int total_blurpts = sum(fx_mask_ != 0); // Count non-zero frequency bands in fx_mask_
        i_pts_.resize(num_bands_, total_blurpts);
        i_pts_.fill(0);
        j_pts_.resize(num_bands_, total_blurpts);
        j_pts_.fill(0);
        ij_ctr_.resize(num_bands_);
        ij_ctr_.fill(0);

        for (uint64_t i = 0; i < fx_mask_.size(0); ++i) {
            for (uint64_t j = 0; j < fx_mask_.size(1); ++j) {
                uint64_t freq_band = fx_mask_(i, j);
                if (freq_band != 0) {
                    uint64_t band_idx = freq_band - 1;
                    i_pts_(band_idx, ij_ctr_(band_idx)) = i;
                    j_pts_(band_idx, ij_ctr_(band_idx)) = j;
                    ij_ctr_(band_idx) += 1;
                }
            }
        }

        fx_mask_band.resize(matrix_size_, matrix_size_, 2, num_bands_);
        fx_mask_band.fill(std::complex<T>(0.0));
        for (uint64_t band = 0; band < num_bands_; ++band) {
            for (uint64_t y = 0; y < matrix_size_; ++y) {
                for (uint64_t x = 0; x < matrix_size_; ++x) {
                    if (fx_mask_(x, y) == band + 1) {
                        fx_mask_band(x, y, 0, band) = std::complex<T>(1.0); 
                        fx_mask_band(x, y, 1, band) = std::complex<T>(1.0); 
                    }
                }
            }
        }
    }

    void blur(const Array<std::complex<T>> &in_image, 
                Array<std::complex<T>> &water_out,
                Array<std::complex<T>> &fat_out,
                const uint16_t te_index = 0, 
                const uint16_t band_index = 0
                ) const
    {
        // It is assumed that water_out and fat_out are already allocated with the correct size.
        auto i_pts_sliced = i_pts_.slice(S(band_index), S::all());
        auto j_pts_sliced = j_pts_.slice(S(band_index), S::all());
        uint64_t ij_ctr_sliced = ij_ctr_(band_index);

        auto te_pha_sliced = te_phase_.slice(S::all(), S::all(), S(te_index), S(band_index));
        auto fmap_residual_sliced = fmap_residual_.slice(S::all(), S::all(), S(band_index));

        // Call the blur kernel function
        do_blur_kern(
            in_image, 
            fmap_residual_sliced, 
            i_pts_sliced, 
            j_pts_sliced, 
            ij_ctr_sliced, 
            water_out, 
            fat_out,        
            kern_table_, 
            kern_radius_, 
            min_index_, 
            te_pha_sliced, 
            freq_inc_,
            band_index
        );
    }

    Array<std::complex<T>> deblur( const Array<std::complex<T>> &water_in,
                                const Array<std::complex<T>> &fat_in,
                                const uint16_t te_index = 0,
                                const uint16_t band_index = 0
                            ) const
    {
        // 1. Prepare the output array. This creates a new, owning array.
        // The dimensions are [matrix_size_, matrix_size_, 2] for water and fat channels.
        Array<std::complex<T>> water_fat_image(matrix_size_, matrix_size_, 2);
        water_fat_image.fill(std::complex<T>(0.0)); // Initialize to zero

        // 2. Slice the input and member arrays to create lightweight views for the deblur kernel.
        // Using 'const auto&' explicitly declares these as non-owning, read-only views.
        const auto te_pha_conj_sliced = te_phase_conj_.slice(S::all(), S::all(), S(te_index), S(band_index));
        const auto fmap_residual_sliced = fmap_residual_.slice(S::all(), S::all(), S(band_index));
        const auto i_pts_sliced = i_pts_.slice(S(band_index), S::all());
        const auto j_pts_sliced = j_pts_.slice(S(band_index), S::all());
        const uint64_t ij_ctr_sliced = ij_ctr_(band_index); // This is a scalar value, not a view.

        // 3. Call the core deblur kernel function.
        // This function is highly optimized as it modifies the 'water_fat_image'
        // array in-place, avoiding any copies of the large data buffer.
        do_deblur_kern(
            water_in,
            fat_in,
            fmap_residual_sliced,
            i_pts_sliced,
            j_pts_sliced,
            ij_ctr_sliced,
            water_fat_image, // Modified in-place
            kern_table_,
            kern_radius_,
            min_index_,
            te_pha_conj_sliced,
            freq_inc_,
            band_index
        );

        // // 4. Apply the mask in-place.
        // // Create a view of the mask for the current band.
        // const auto fx_mask_band_sliced = fx_mask_band.slice(S::all(), S::all(), S::all(), S(band_index));

        // // Perform element-wise multiplication and assignment in-place.
        // // The `*=` operator is optimized to handle strides and does not
        // // create a temporary array for the product, maximizing performance.
        // water_fat_image *= fx_mask_band_sliced;
        
        // 5. Return the result using move semantics.
        // This avoids a costly deep copy of the 'water_fat_image' array
        // when returning from the function. Instead, its ownership is efficiently
        // transferred to the caller.
        return std::move(water_fat_image);
    }

    uint64_t num_te() const {
        return num_te_;
    }

    uint64_t num_bands() const {
        return num_bands_;
    }

    const Array<uint64_t>& i_pts() const { return i_pts_; }
    const Array<uint64_t>& j_pts() const { return j_pts_; }
    const Array<uint64_t>& ij_ctr() const { return ij_ctr_; }

    const Array<uint64_t>& get_fx_mask() const { return fx_mask_; }

};
}
#endif // SVCONV_HPP
