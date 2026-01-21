#ifndef GRIDDER_HPP
#define GRIDDER_HPP

#include "GPIArray/GPIArray.hpp"
#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <algorithm>

constexpr int KERNSIZE = 250;
constexpr int KERNEL_SCALE = 100;
constexpr double PI = 3.14159265358979323846;

using namespace GPIArray;

namespace MRIRecon {
template <typename T>
class Gridder {
private:
    Array<T> x_coords_;
    Array<T> y_coords_;
    Array<T> dcf_weights_;
    uint64_t matrix_size_;
    uint64_t num_samples_;
    float ovs_factor_ = 1.5f; // Oversampling factor
    uint64_t osmtx_size_; // Oversampled matrix size
    int num_dsets_ = 1; // Number of datasets (default to 1)

    Array<T> kernel_; // Kernel array
    Array<T> pr_mask_; // Precomputed rolloff mask

    std::shared_ptr<FFTW::FFTPlanManager<T>> fft2d_plan;

    //========================================================================
    // ROLLOFFMASK
    //========================================================================

    void gen_rolloffmask()
    {
        // Initialize kernel array
        kernel_.resize(2 * KERNSIZE + 1);
        kernel_.fill(0.0);

        // Generate symmetric kernel: Hanning window times exp(-6 x^2)
        for (int i = 0; i <= KERNSIZE; ++i) {
            T x = static_cast<T>(i) / KERNSIZE;
            T val = 0.5 * (1.0 + std::cos(PI * x)) * std::exp(-6.0 * x * x);
            kernel_(KERNSIZE + i) = val;
            kernel_(KERNSIZE - i) = val;
        }

        int kernwidth = KERNSIZE / 100;
        Array<std::complex<T>> roll(osmtx_size_);
        roll.fill(0.0);

        // Fill roll array with kernel values
        for (int i = -kernwidth; i <= kernwidth; ++i) {
            int idx = (osmtx_size_ / 2) + i;
            if (idx >= 0 && idx < static_cast<int>(osmtx_size_))
                roll(idx) = kernel_(KERNSIZE + (100 * i));
        }

        FFTW::fft1(roll, roll, FFTW_BACKWARD);

        int di_r = (osmtx_size_ / 2) - (matrix_size_ / 2);
        Array<T> corr(matrix_size_);
        std::complex<T> rmid = roll(osmtx_size_ / 2);

        for (uint64_t i = 0; i < matrix_size_; ++i) {
            corr(i) = std::abs(rmid / roll(i + di_r));
        }

        // Compute rolloff mask as outer product(corr, corr)
        pr_mask_ = outer_product(corr, corr);
    }
    
    // Template version of griddat to support complex<float> or complex<double>
    void griddat(const Array<T> &xcrds, const Array<T> &ycrds, const Array<std::complex<T>> &spdata,
                const Array<T> &kernel, const Array<T> &pr_mask, const Array<T> &wates,
                Array<std::complex<T>> &outdata, long osmtx) const
    {
        uint64_t d;
        T fmtx = static_cast<T>(osmtx);
        T mtxd2 = fmtx / 2;
        T krad = 0.01 * static_cast<T>(KERNSIZE);

        Array<std::complex<T>> zmtx(osmtx, osmtx);
        zmtx.fill(std::complex<T>(0));
        outdata.fill(std::complex<T>(0));

        for (d = 0; d < spdata.size(); ++d) {
            std::complex<T> val0 = spdata(d) * wates(d);
            T fi = mtxd2 + (xcrds(d) * fmtx);
            T fj = mtxd2 + (ycrds(d) * fmtx);

            uint64_t mini = static_cast<uint64_t>(std::max(T(0), std::ceil(fi - krad)));
            uint64_t maxi = static_cast<uint64_t>(std::min(std::floor(fi + krad), fmtx - 1));
            uint64_t minj = static_cast<uint64_t>(std::max(T(0), std::ceil(fj - krad)));
            uint64_t maxj = static_cast<uint64_t>(std::min(std::floor(fj + krad), fmtx - 1));

            // Swapped loops for better cache locality (innermost loop over columns 'j')
            for (uint64_t i = mini; i <= maxi; ++i) { // Outer loop: iterates over rows
                int current_di_idx = KERNSIZE + static_cast<int>(std::floor(100.0 * (static_cast<T>(i) - fi) + 0.5));
                T kern_i_val = kernel(current_di_idx); // Precompute kernel(di) for this row

                for (uint64_t j = minj; j <= maxj; ++j) { // Inner loop: iterates over columns (contiguous access)
                    int current_dj_idx = KERNSIZE + static_cast<int>(std::floor(100.0 * (static_cast<T>(j) - fj) + 0.5));
                    T kern_j_val = kernel(current_dj_idx); // Precompute kernel(dj) for this column

                    zmtx(i, j) += kern_i_val * kern_j_val * val0;
                }
            }
        }

        fft2d_plan->execute_backward(zmtx);

        uint64_t di_zp = (osmtx / 2) - (outdata.size(0) / 2);
        for (uint64_t i = 0; i < outdata.size(0); ++i) {
            for (uint64_t j = 0; j < outdata.size(0); ++j) {
                outdata(i, j) = zmtx(i + di_zp, j + di_zp) * pr_mask(i, j);
            }
        }
    }

    void ungriddat(const Array<T> &xcrds, const Array<T> &ycrds, const Array<std::complex<T> > &imdata,
              Array<std::complex<T> > &spdata, const Array<T> &kernel,
              const Array<T> &pr_mask, const long osmtx) const
    {
        uint64_t d, di_zp;
        T fmtx = static_cast<T>(osmtx);
        T mtxd2 = fmtx / 2;
        T krad = 0.01 * static_cast<T>(KERNSIZE);

        Array<std::complex<T>> zmtx(osmtx, osmtx);
        zmtx.fill(std::complex<T>(0));
        spdata.fill(std::complex<T>(0));

        di_zp = (osmtx / 2) - (imdata.size(0) / 2);
        for (uint64_t i = 0; i < imdata.size(0); ++i) {
            for (uint64_t j = 0; j < imdata.size(0); ++j) {
            zmtx(i + di_zp, j + di_zp) = pr_mask(i, j) * imdata(i, j);
            }
        }

        fft2d_plan->execute_forward(zmtx);

        for (d = 0; d < spdata.size(); ++d) {
            T fi = mtxd2 + (xcrds(d) * fmtx);
            T fj = mtxd2 + (ycrds(d) * fmtx);

            uint64_t mini = static_cast<uint64_t>(std::max(T(0), std::ceil(fi - krad)));
            uint64_t maxi = static_cast<uint64_t>(std::min(std::floor(fi + krad), fmtx - 1));
            uint64_t minj = static_cast<uint64_t>(std::max(T(0), std::ceil(fj - krad)));
            uint64_t maxj = static_cast<uint64_t>(std::min(std::floor(fj + krad), fmtx - 1));

            // Swapped loops for better cache locality (innermost loop over columns 'j')
            std::complex<T> val2(0.0, 0.0);
            for (uint64_t i = mini; i <= maxi; ++i) { // Outer loop: iterate over rows
                int current_di_idx = KERNSIZE + static_cast<int>(std::floor(100.0 * (static_cast<T>(i) - fi) + 0.5));
                T kern_i_val = kernel(current_di_idx); // Precompute kernel(di) for this row

                std::complex<T> val1(0.0, 0.0);
                for (uint64_t j = minj; j <= maxj; ++j) { // Inner loop: iterates over columns (contiguous access)
                    int current_dj_idx = KERNSIZE + static_cast<int>(std::floor(100.0 * (static_cast<T>(j) - fj) + 0.5));
                    T kern_j_val = kernel(current_dj_idx); // Precompute kernel(dj) for this column

                    val1 += zmtx(i, j) * kern_j_val; // Use kern_j_val with zmtx(i,j) for contiguous access
                }
                val2 += val1 * kern_i_val; // Use kern_i_val with val1
            }
            spdata(d) = val2;
        }
    }

public:

Gridder() = default; // Default constructor initializes empty gridder

Gridder(const Array<T>& x_coords, const Array<T>& y_coords, const Array<T>& dcf_weights, uint64_t matrix_size, float ovs_factor = 1.5f)
    : x_coords_(x_coords),
      y_coords_(y_coords),
      dcf_weights_(dcf_weights),
      matrix_size_(matrix_size),
      num_samples_(x_coords.size(0)),
      ovs_factor_(ovs_factor),
      osmtx_size_(static_cast<uint64_t>(std::ceil(matrix_size * ovs_factor)))
{
    if (x_coords_.size() != y_coords_.size()) {
        throw std::invalid_argument("Gridder: 'x_coords' and 'y_coords' must be 1D arrays of the same size.");
    }
    if (dcf_weights_.size(0) != num_samples_) {
        throw std::invalid_argument("Gridder: 'dcf_weights' must be 1D and match the number of samples.");
    }
    

    if(x_coords_.ndim() > 1)
        num_dsets_ = x_coords_.size(1); // Assume first dimension is number of datasets
    
    if (ovs_factor_ < 1.0f || ovs_factor_ >= 5.0f) {
        throw std::invalid_argument("Gridder: 'ovs_factor' must be >= 1 and < 5.");
    }
    gen_rolloffmask();

    fft2d_plan = std::make_shared<FFTW::FFTPlanManager<T>>(std::vector<uint64_t>{osmtx_size_, osmtx_size_}, FFTW_MEASURE);
}   

// Gridder destructor
~Gridder() {
    // No explicit cleanup needed, smart pointers handle memory
}


Array<std::complex<T>> ksp_img(const Array<std::complex<T>>& ksp, const int dset_idx = -1,
    bool parallel_thread = false) const {

    // --- Input Validation ---
    auto ksp_shape = ksp.dimensions_vector();

    if (ksp_shape[0] != num_samples_) {
        throw std::invalid_argument("Gridder: 0th axis of ksp input must match num_samples_.");
    }

    // --- FAST PATH for simple 1D input (minimal overhead) ---
    // If ksp is 1D and Gridder is configured for a single dataset, process directly.
    if (ksp.ndim() == 1 && num_dsets_ == 1) {
        Array<std::complex<T>> outdata(matrix_size_, matrix_size_);
        griddat(
            x_coords_, y_coords_, ksp,
            kernel_, pr_mask_, dcf_weights_,
            outdata, osmtx_size_
        );
        return std::move(outdata);
    }
    else if(ksp.ndim() == 1 && num_dsets_ > 1 && dset_idx >= 0 && dset_idx < num_dsets_) {
        // If ksp is 1D and Gridder is configured for multiple datasets, process the specified dataset.
        Array<std::complex<T>> outdata(matrix_size_, matrix_size_);
        griddat(
            x_coords_.slice(S::all(), S(dset_idx)), 
            y_coords_.slice(S::all(), S(dset_idx)), 
            ksp,
            kernel_, pr_mask_, 
            dcf_weights_.slice(S::all(), S(dset_idx)),
            outdata, osmtx_size_
        );
        return std::move(outdata);
    }
    // --- END FAST PATH ---


    // Determine output image dimensions
    std::vector<uint64_t> outdata_dims;
    outdata_dims.push_back(matrix_size_);
    outdata_dims.push_back(matrix_size_);

    uint64_t ksp_extra_dims_start_idx;
    uint64_t num_current_dsets_in_ksp_input = 1;

    // Logic for handling dataset dimension and extra dimensions
    if (num_dsets_ > 1) { // Gridder is configured for multiple datasets
        // if (ksp_shape.size() < 2 || ksp_shape[1] != num_dsets_) {
        //     throw std::invalid_argument("Gridder: ksp input must have its 1st dimension matching num_dsets_ when gridder is multi-dataset.");
        // }
        num_current_dsets_in_ksp_input = num_dsets_;
        ksp_extra_dims_start_idx = 2; // Extra dims start from the 3rd dimension of ksp (index 2)
        outdata_dims.push_back(num_dsets_); // Add the dataset dimension to output
    } else { // Gridder is configured for a single dataset (num_dsets_ == 1)
        ksp_extra_dims_start_idx = 1; // Extra dims start from the 2nd dimension of ksp (index 1)
    }

    // Append any extra dimensions from the ksp input to the output image dimensions
    if (ksp_shape.size() > ksp_extra_dims_start_idx) {
        for (uint64_t i = ksp_extra_dims_start_idx; i < ksp_shape.size(); ++i) {
            outdata_dims.push_back(ksp_shape[i]);
        }
    }

    // Create the output Array with the determined dimensions
    Array<std::complex<T>> outdata(outdata_dims);
    outdata.fill(std::complex<T>(0)); // Initialize output to zeros


    // Determine the loop range for datasets/extra dimensions
    std::vector<uint64_t> iteration_dims; // Dimensions over which to loop for gridding calls
    if (num_current_dsets_in_ksp_input > 1) {
        iteration_dims.push_back(num_current_dsets_in_ksp_input); // Dataset dimension
    }
    for (uint64_t i = ksp_extra_dims_start_idx; i < ksp_shape.size(); ++i) {
        iteration_dims.push_back(ksp_shape[i]); // Extra dimensions
    }
    
    // Calculate total number of gridding iterations
    uint64_t total_gridding_iterations = 1;
    for(uint64_t dim : iteration_dims) {
        total_gridding_iterations *= dim;
    }

    // Handle single dataset case (no explicit looping over datasets/extra dims)
    if (total_gridding_iterations == 0) {
        return std::move(outdata);
    }
    
    // Define the lambda function for gridding a single logical "slice" of data
    // This lambda captures 'this' (Gridder object), ksp_shape, ksp_extra_dims_start_idx,
    // num_dsets_, outdata_dims, and parallel_thread.
    auto grid_single_slice_lambda =
        [&](uint64_t iter_idx) {
        
        // Reconstruct N-dimensional indices for ksp, x_coords, y_coords, dcf_weights, and outdata
        // from the flat iter_idx.
        std::vector<uint64_t> full_ksp_input_indices(ksp_shape.size());
        full_ksp_input_indices[0] = 0; // The sample index will always be S::all()

        // Unravel iter_idx to get the indices for datasets and extra dimensions
        uint64_t temp_iter_idx_for_unravel = iter_idx;
        uint64_t logical_dim_counter = 0;
        
        if (num_dsets_ > 1) { // If dataset dimension exists in ksp and coords
            full_ksp_input_indices[1] = temp_iter_idx_for_unravel % iteration_dims[logical_dim_counter];
            temp_iter_idx_for_unravel /= iteration_dims[logical_dim_counter];
            logical_dim_counter++;
        }

        for (uint64_t d = ksp_extra_dims_start_idx; d < ksp_shape.size(); ++d) {
            full_ksp_input_indices[d] = temp_iter_idx_for_unravel % iteration_dims[logical_dim_counter];
            temp_iter_idx_for_unravel /= iteration_dims[logical_dim_counter];
            logical_dim_counter++;
        }

        // Construct slice objects or indices for the current gridding call
        std::vector<Slice> current_ksp_slices;
        current_ksp_slices.push_back(S::all()); // For samples
        for (uint64_t d = 1; d < ksp_shape.size(); ++d) { // For datasets and extra dims
            current_ksp_slices.push_back(S(full_ksp_input_indices[d]));
        }
        auto ksp_slice = ksp.slice(current_ksp_slices);

        // x_coords_slice, y_coords_slice, dcf_weights_slice use same indices but map to their respective dimensions
        std::vector<Slice> current_coords_slices;
        current_coords_slices.push_back(S::all()); // For samples
        for (uint64_t d = 1; d < x_coords_.ndim(); ++d) { // For datasets and extra dims (assuming same dimensionality as ksp for these)
            current_coords_slices.push_back(S(full_ksp_input_indices[d]));
        }
        auto x_coords_slice = x_coords_.slice(current_coords_slices);
        auto y_coords_slice = y_coords_.slice(current_coords_slices);
        auto dcf_weights_slice = dcf_weights_.slice(current_coords_slices);


        // outdata_slice: (S::all(), S::all(), full_ksp_input_indices[1], full_ksp_input_indices[2], ...)
        std::vector<Slice> current_outdata_slices;
        current_outdata_slices.push_back(S::all()); // For matrix_size (row)
        current_outdata_slices.push_back(S::all()); // For matrix_size (col)
        for (uint64_t d = 1; d < outdata_dims.size() - 1; ++d) { // For datasets and extra dims (offset by 2 for matrix_size)
            current_outdata_slices.push_back(S(full_ksp_input_indices[d]));
        }
        auto outdata_slice = outdata.slice(current_outdata_slices);


        // --- Actual Gridding Call ---
        // Pass the parallel_thread_inner_loops control to griddat.
        this->griddat( // Use this->griddat to call the member function
            x_coords_slice, y_coords_slice, ksp_slice,
            kernel_, pr_mask_, dcf_weights_slice,
            outdata_slice, osmtx_size_
        );
    };

    // --- Parallelize gridding over datasets/extra dimensions ---
    if (parallel_thread) {
        #pragma omp parallel for
        for (uint64_t iter_idx = 0; iter_idx < total_gridding_iterations; ++iter_idx) {
            grid_single_slice_lambda(iter_idx);
        }
    } else {
        for (uint64_t iter_idx = 0; iter_idx < total_gridding_iterations; ++iter_idx) {
            grid_single_slice_lambda(iter_idx);
        }
    }

    //std::string fname = "/Users/gkrishnamoo3/Documents/Data/Exam9898_05Jun2025_vvol_composite/debug1.npy";
    //Numpy::write_npy(ksp, fname); // Debugging output
    
    return std::move(outdata);

}

// Corrected snippet for img_ksp
Array<std::complex<T>> img_ksp(const Array<std::complex<T>>& in_image, const int dset_idx = -1,
     bool parallel_thread = false) const {
    
    // --- Input Validation ---
    auto in_image_shape = in_image.dimensions_vector();

    if (in_image_shape.size() < 2 || in_image_shape[0] != matrix_size_ || in_image_shape[1] != matrix_size_) {
        throw std::invalid_argument("Gridder: in_image must have at least 2 dimensions, with first two matching matrix_size_.");
    }

    // --- FAST PATH for simple 2D input (minimal overhead) ---
    // If in_image is 2D and Gridder is configured for a single dataset, process directly.
    if (in_image.ndim() == 2 && num_dsets_ == 1) {
        Array<std::complex<T>> out_spdata(num_samples_);
        // Removed parallel_thread arg from ungriddat call
        ungriddat(x_coords_, y_coords_, in_image,
              out_spdata, kernel_,
              pr_mask_, osmtx_size_
        );
        return std::move(out_spdata);
    }
    else if (in_image.ndim() == 2 && num_dsets_ > 1 && dset_idx >= 0 && dset_idx < num_dsets_) {
        Array<std::complex<T>> out_spdata(num_samples_);
        ungriddat(
            x_coords_.slice(S::all(), S(dset_idx)),
            y_coords_.slice(S::all(), S(dset_idx)),
            in_image,
            out_spdata, kernel_,
            pr_mask_, osmtx_size_
        );
        return std::move(out_spdata);
    }
    // --- END FAST PATH ---

    // --- Determine output k-space data dimensions ---
    std::vector<uint64_t> out_spdata_dims;
    out_spdata_dims.push_back(num_samples_);

    uint64_t img_extra_dims_start_idx;
    uint64_t num_current_dsets_in_img_input = 1;

    // Logic for handling dataset dimension and extra dimensions
    if (num_dsets_ > 1) { // Gridder is configured for multiple datasets
        if (in_image_shape.size() < 3 || in_image_shape[2] != static_cast<uint64_t>(num_dsets_)) {
            throw std::invalid_argument("Gridder: in_image input must have its 3rd dimension matching num_dsets_ when gridder is multi-dataset.");
        }
        num_current_dsets_in_img_input = num_dsets_;
        img_extra_dims_start_idx = 3; // Extra dims start from the 4th dimension of in_image (index 3)
        out_spdata_dims.push_back(num_dsets_);
    } else {
        img_extra_dims_start_idx = 2; // Extra dims start from the 3rd dimension of in_image (index 2)
    }

    // Append any extra dimensions from the in_image input to the output k-space data dimensions
    if (in_image_shape.size() > img_extra_dims_start_idx) {
        for (uint64_t i = img_extra_dims_start_idx; i < in_image_shape.size(); ++i) {
            out_spdata_dims.push_back(in_image_shape[i]);
        }
    }

    // Create the output Array with the determined dimensions
    Array<std::complex<T>> out_spdata(out_spdata_dims);
    out_spdata.fill(std::complex<T>(0));

    // Determine the loop range for datasets/extra dimensions
    std::vector<uint64_t> iteration_dims;
    if (num_current_dsets_in_img_input > 1) {
        iteration_dims.push_back(num_current_dsets_in_img_input);
    }
    for (uint64_t i = img_extra_dims_start_idx; i < in_image_shape.size(); ++i) {
        iteration_dims.push_back(in_image_shape[i]);
    }

    uint64_t total_ungridding_iterations = 1;
    for (uint64_t dim : iteration_dims) {
        total_ungridding_iterations *= dim;
    }

    if (total_ungridding_iterations == 0) {
        return std::move(out_spdata);
    }

    auto ungrid_single_slice_lambda =
        [&](uint64_t iter_idx) {
        std::vector<uint64_t> full_img_input_indices(in_image_shape.size());
        full_img_input_indices[0] = 0; // row
        full_img_input_indices[1] = 0; // col

        uint64_t temp_iter_idx_for_unravel = iter_idx;
        uint64_t logical_dim_counter = 0;

        if (num_dsets_ > 1) {
            full_img_input_indices[2] = temp_iter_idx_for_unravel % iteration_dims[logical_dim_counter];
            temp_iter_idx_for_unravel /= iteration_dims[logical_dim_counter];
            logical_dim_counter++;
        }
        for (uint64_t d = img_extra_dims_start_idx; d < in_image_shape.size(); ++d) {
            full_img_input_indices[d] = temp_iter_idx_for_unravel % iteration_dims[logical_dim_counter];
            temp_iter_idx_for_unravel /= iteration_dims[logical_dim_counter];
            logical_dim_counter++;
        }

        // Slices for in_image
        std::vector<Slice> current_img_slices;
        current_img_slices.push_back(S::all());
        current_img_slices.push_back(S::all());
        for (uint64_t d = 2; d < in_image_shape.size(); ++d) {
            current_img_slices.push_back(S(full_img_input_indices[d]));
        }
        auto in_image_slice = in_image.slice(current_img_slices);

        // Slices for coords/weights
        std::vector<Slice> current_coords_slices;
        current_coords_slices.push_back(S::all());
        for (uint64_t d = 1; d < x_coords_.ndim(); ++d) {
            current_coords_slices.push_back(S(full_img_input_indices[d+1]));
        }
        auto x_coords_slice = x_coords_.slice(current_coords_slices);
        auto y_coords_slice = y_coords_.slice(current_coords_slices);
        auto dcf_weights_slice = dcf_weights_.slice(current_coords_slices);

        // Slices for out_spdata
        std::vector<Slice> current_out_spdata_slices;
        current_out_spdata_slices.push_back(S::all());
        for (uint64_t d = 1; d < out_spdata_dims.size(); ++d) {
            current_out_spdata_slices.push_back(S(full_img_input_indices[d+1]));
        }
        auto out_spdata_slice = out_spdata.slice(current_out_spdata_slices);

        this->ungriddat(
            x_coords_slice, y_coords_slice, in_image_slice,
            out_spdata_slice, kernel_,
            pr_mask_, osmtx_size_
        );
    };

    if (parallel_thread) {
        #pragma omp parallel for
        for (uint64_t iter_idx = 0; iter_idx < total_ungridding_iterations; ++iter_idx) {
            ungrid_single_slice_lambda(iter_idx);
        }
    } else {
        for (uint64_t iter_idx = 0; iter_idx < total_ungridding_iterations; ++iter_idx) {
            ungrid_single_slice_lambda(iter_idx);
        }
    }

    return std::move(out_spdata);
    
}

uint64_t num_samples() const {
    return num_samples_;
}

void test_case(const Array<float>& input)
{
    for (uint64_t i = 0; i < input.size(); ++i) {
        std::cout << "input[" << i << "] = " << input(i) << std::endl;
    }

}


};
} // namespace MRIRecon

#endif // GRIDDER_HPP
