#ifndef DEBLUR_FX_HPP
#define DEBLUR_FX_HPP

#include "GPIArray/GPIArray.hpp"
#include "DeblurFXArgs.hpp"
#include "Gridder.hpp"
#include "SpatVarConv.hpp"      
#include "PhaseModulator.hpp"

using namespace GPIArray;

namespace MRIRecon {


template <typename T>
class DeblurFX {
    // Compile-time check for supported real types for complex numbers
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "DeblurFX class only supports std::complex<double> or std::complex<float> for its numeric type T.");

private:
    // Member variables as direct objects (no pointers)
    Gridder<T> grid_op;
    Gridder<T> grid_cg_op; // This needs careful handling for the conditional construction
    PhaseModulator<T> phase_mod_op;
    SpatVarConv<T> spat_conv_op;

    Array<std::complex<T>> csm;
    Array<std::complex<T>> csm_conj; // Conjugate of csm, if needed

    uint64_t num_te = 0; // Number of echo times
    uint64_t num_bands = 0; // Number of bands
    uint64_t num_coils = 0; // Number of coils
    uint64_t matrix_size = 0; // Matrix size, assuming square matrix
    uint64_t num_samples = 0; // Number of samples in k-space
    uint64_t num_samples_cg = 0; // Number of samples in k-space for CG

    DeblurArgs<T> deb_args;
    GridArgs<T> grid_args;

public:

DeblurFX() = default;

DeblurFX(const Gridder<T>& grid_op_in,
        const Gridder<T>& grid_cg_op_in,
        const SpatVarConv<T>& spat_conv_op_in,
        const PhaseModulator<T>& phase_mod_op_in,
         const Array<std::complex<T>>& csm_in
) : grid_op(grid_op_in),
    grid_cg_op(grid_cg_op_in),
    phase_mod_op(phase_mod_op_in),
    spat_conv_op(spat_conv_op_in),
    csm(csm_in),
    csm_conj(conj(csm_in)) // Compute conjugate of csm once
    {
        num_te = spat_conv_op.num_te(); // Number of echo times
        num_bands = spat_conv_op.num_bands(); // Number of bands
        num_coils = csm.size(2); // Number of coils
        matrix_size = csm.size(0); // Matrix size, assuming square matrix
        num_samples = grid_op.num_samples(); // Number of samples in k-space
        num_samples_cg = grid_cg_op.num_samples(); // Number of samples in k-space for CGß
    }

Array<std::complex<T>> adjoint_op(const Array<std::complex<T>>& ksp_data) const
{
    Array<std::complex<T>> water_buff(matrix_size, matrix_size);
    Array<std::complex<T>> fat_buff(matrix_size, matrix_size);
    
    Array<std::complex<T>> water_fat_image(matrix_size, matrix_size, 2);
    water_fat_image.fill(std::complex<T>(0.0));

    Array<std::complex<T>> water_coilcombined(matrix_size, matrix_size, num_te);
    Array<std::complex<T>> fat_coilcombined(matrix_size, matrix_size, num_te);
    Array<std::complex<T>> img_slice;
    
    for (uint64_t band = 0; band < num_bands; ++band) {
        water_coilcombined.fill(std::complex<T>(0.0));
        fat_coilcombined.fill(std::complex<T>(0.0));

        for (uint64_t te = 0; te < num_te; ++te) {
            auto water_slice = water_coilcombined.slice(S::all(), S::all(), S(te));
            auto fat_slice = fat_coilcombined.slice(S::all(), S::all(), S(te));

            for (uint64_t coil = 0; coil < num_coils; ++coil) {
                auto ksp_slice = ksp_data.slice(S::all(), S(te), S(coil));
                
                if( ksp_slice.size(0) == num_samples_cg ) {
                    img_slice = grid_cg_op.ksp_img(ksp_slice, te);
                }
                else{
                    img_slice = grid_op.ksp_img(ksp_slice, te);
                }
                
                phase_mod_op.demodulate_phase(img_slice, water_buff, fat_buff, te, band);

                for (uint64_t j = 0; j < matrix_size; ++j) {
                    for (uint64_t i = 0; i < matrix_size; ++i) {
                        water_slice(i, j) += water_buff(i, j) * csm_conj(i, j, coil);
                        fat_slice(i, j) += fat_buff(i, j) * csm_conj(i, j, coil);
                    }
                }
            }
            water_fat_image += spat_conv_op.deblur(water_slice, fat_slice, te, band);
        }
    }

    return std::move(water_fat_image);
}

Array<std::complex<T>> forward_op(const Array<std::complex<T>>& img_data) const
{
    Array<std::complex<T>> water_buff(matrix_size, matrix_size, num_bands);
    Array<std::complex<T>> fat_buff(matrix_size, matrix_size, num_bands);

    Array<std::complex<T>> img_te_coil_buff(matrix_size, matrix_size);
    

    Array<std::complex<T>> ksp_out(num_samples, num_te, num_coils);
    ksp_out.fill(std::complex<T>(0.0));

    for (uint64_t te = 0; te < num_te; ++te) {
        for (uint64_t band = 0; band < num_bands; ++band) {
            auto water_buff_slice = water_buff.slice(S::all(), S::all(), S(band));
            auto fat_buff_slice = fat_buff.slice(S::all(), S::all(), S(band));
            spat_conv_op.blur(img_data, water_buff_slice, fat_buff_slice, te, band);
        }

        for (uint64_t coil = 0; coil < num_coils; ++coil) {
            img_te_coil_buff.fill(std::complex<T>(0.0));
            for (uint64_t band = 0; band < num_bands; ++band) {
                auto water_buff_slice = water_buff.slice(S::all(), S::all(), S(band)).copy();
                auto fat_buff_slice = fat_buff.slice(S::all(), S::all(), S(band)).copy();

                for (uint64_t i = 0; i < matrix_size; ++i) {
                    for (uint64_t j = 0; j < matrix_size; ++j) {
                        water_buff_slice(i, j) *= csm(i, j, coil);
                        fat_buff_slice(i, j) *= csm(i, j, coil);
                    }
                }

                img_te_coil_buff += phase_mod_op.modulate_phase(water_buff_slice, fat_buff_slice, te, band);

            }
            auto ksp_out_slice = ksp_out.slice(S::all(), S(te), S(coil));
            ksp_out_slice += grid_op.img_ksp(img_te_coil_buff, te);
        }
    }
    return std::move(ksp_out);
}

uint64_t get_matrix_size() const {
    return matrix_size;
}

};
} // namespace MRIRecon

#endif // DEBLUR_FX_HPP