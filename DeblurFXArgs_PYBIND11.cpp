#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // For std::vector, etc.
#include <pybind11/complex.h>      // For std::complex support
#include <pybind11/operators.h>    // For operator overloads if needed (e.g., for structs)

#include "src/DeblurFXArgs.hpp" // Include the consolidated args file

namespace py = pybind11;
using namespace GPIArray; // Use GPIArray namespace

namespace MRIRecon {

// Helper function to bind DeblurArgs for a specific type T
template <typename T>
void bind_deblur_args(py::module_ &m, const char *name_suffix) {
    py::class_<DeblurArgs<T>>(m, ("DeblurArgs" + std::string(name_suffix)).c_str(), py::module_local())
        .def(py::init<>()) // Default constructor
        .def_readwrite("fmap_residual", &DeblurArgs<T>::fmap_residual)
        .def_readwrite("fx_mask", &DeblurArgs<T>::fx_mask)
        .def_readwrite("ksp_phase", &DeblurArgs<T>::ksp_phase)
        .def_readwrite("te_phase", &DeblurArgs<T>::te_phase)
        .def_readwrite("kern_table", &DeblurArgs<T>::kern_table)
        .def_readwrite("kern_radius", &DeblurArgs<T>::kern_radius)
        .def_readwrite("minindex", &DeblurArgs<T>::minindex)
        .def_readwrite("freq_increment", &DeblurArgs<T>::freq_increment);
}

// Helper function to bind GridArgs for a specific type T
template <typename T>
void bind_grid_args(py::module_ &m, const char *name_suffix) {
    py::class_<GridArgs<T>>(m, ("GridArgs" + std::string(name_suffix)).c_str(), py::module_local())
        .def(py::init<>()) // Default constructor
        .def_readwrite("coords", &GridArgs<T>::coords)
        .def_readwrite("sdc", &GridArgs<T>::sdc)
        .def_readwrite("coords_cg", &GridArgs<T>::coords_cg)
        .def_readwrite("sdc_cg", &GridArgs<T>::sdc_cg)
        .def_readwrite("matrix_size", &GridArgs<T>::matrix_size)
        .def_readwrite("oversample_factor", &GridArgs<T>::oversample_factor);
}

// Helper function to bind OptimizationArgs for a specific type T
template <typename T>
void bind_cg_optimization_args(py::module_ &m, const char *name_suffix) {
    // Bind the enum OptimizationMethod
    // Bind the enum OptimizationMethod as an int property
    py::class_<OptimizationArgs<T>>(m, ("OptimizationArgs" + std::string(name_suffix)).c_str(), py::module_local())
        .def(py::init<>())
        .def_readwrite("lambda_reg", &OptimizationArgs<T>::lambda_reg)
        .def_readwrite("max_iterations", &OptimizationArgs<T>::max_iterations)
        .def_readwrite("tolerance", &OptimizationArgs<T>::tolerance)
        .def_readwrite("wavelet_levels", &OptimizationArgs<T>::wavelet_levels)
        .def_readwrite("use_d2_not_d4", &OptimizationArgs<T>::use_d2_not_d4)
        .def_readwrite("update_field_map", &OptimizationArgs<T>::update_field_map)
        .def_readwrite("echo_times", &OptimizationArgs<T>::echo_times)
        .def_readwrite("fat_peaks", &OptimizationArgs<T>::fat_peaks)
        .def_readwrite("field_map_damping_factor", &OptimizationArgs<T>::field_map_damping_factor)
        .def_property("optimization_method",
            // GETTER: Returns the enum value as int
            [](const OptimizationArgs<T> &self) -> int {
                return static_cast<int>(self.optimization_method);
            },
            // SETTER: Sets the enum value from int
            [](OptimizationArgs<T> &self, int method_int) {
                using Method = typename OptimizationArgs<T>::OptimizationMethod;
                switch (method_int) {
                    case static_cast<int>(Method::CG):
                        self.optimization_method = Method::CG;
                        break;
                    case static_cast<int>(Method::FISTA):
                        self.optimization_method = Method::FISTA;
                        break;
                    case static_cast<int>(Method::ADMM):
                        self.optimization_method = Method::ADMM;
                        break;
                    default:
                        throw std::invalid_argument("Unknown optimization method int: " + std::to_string(method_int));
                }
            });
}

} // namespace MRIRecon

PYBIND11_MODULE(DeblurFXArgs, m) {
    m.doc() = "Pybind11 binding for MRIRecon argument structs.";

    // Bind for float precision
    MRIRecon::bind_deblur_args<float>(m, "F");
    MRIRecon::bind_grid_args<float>(m, "F");
    MRIRecon::bind_cg_optimization_args<float>(m, "F");

    // Bind for double precision
    MRIRecon::bind_deblur_args<double>(m, "D");
    MRIRecon::bind_grid_args<double>(m, "D");
    MRIRecon::bind_cg_optimization_args<double>(m, "D");
}
