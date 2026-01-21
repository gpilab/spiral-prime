#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // For std::vector, etc.
#include <pybind11/complex.h>      // For std::complex support
#include <pybind11/operators.h>    // For operator overloads if needed (e.g., for structs)

#include "src/CGMultiSlice.hpp"

namespace py = pybind11;
using namespace GPIArray; // Use GPIArray namespace

namespace MRIRecon {

// Helper function to bind CGTikhonovSolver for a specific type T
template <typename T>
void bind_cg_multislice_solver(py::module_ &m, const char *name_suffix) {
    using Solver = CGMultiSlice<T>;
    std::string class_name = std::string("CGMultiSlice") + name_suffix;

    py::class_<Solver>(m, class_name.c_str())
        .def(py::init<const Array<std::complex<T>> &,
                      const Array<std::complex<T>> &,
                      const DeblurArgs<T> &,
                      const GridArgs<T> &>(),
             py::arg("ksp_in"),
             py::arg("csm_in"),
             py::arg("deb_args1"),
             py::arg("grid_args1"))
        .def(py::init<const Array<std::complex<T>> &,
                      const Array<std::complex<T>> &,
                      const DeblurArgs<T> &,
                      const GridArgs<T> &,
                      const Array<std::complex<T>> &,
                      const DeblurArgs<T> &,
                      const GridArgs<T> &>(),
             py::arg("ksp_in"),
             py::arg("csm_in"),
             py::arg("deb_args1"),
             py::arg("grid_args1"),
             py::arg("ksp_out"),
             py::arg("deb_args_out"),
             py::arg("grid_args_out"))
        .def("get_initial_image", &Solver::get_initial_image,
                "Return the initial image as an array.")
        .def("get_bg_mask", &Solver::get_bg_mask,
                "Return the background mask as an array of bools.")
        .def("cg_solve", 
             [](Solver &self, const Array<std::complex<T>> &curr_image, T rho, uint64_t max_iterations) {
                 return self.cg_solve(curr_image, rho, max_iterations);
             },
             py::arg("curr_image"),
             py::arg("rho"),
             py::arg("max_iterations"),
             "Solve the CGMultiSlice problem and return the result as an array.");
}

} // namespace MRIRecon

PYBIND11_MODULE(CGMultiSlice, m) {
    m.doc() = "Pybind11 binding for MRIRecon::CGMultiSlice C++ class.";

    // Bind for float precision
    MRIRecon::bind_cg_multislice_solver<float>(m, "F");

    // Bind for double precision
    MRIRecon::bind_cg_multislice_solver<double>(m, "D");
}
