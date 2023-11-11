#include <pybind11/pybind11.h>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#include "model.h"


namespace py = pybind11;
using std::vector;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: ising

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";
    py::class_<Model>(m, "Model")
        .def(py::init<Real, Real, vector<int>>())
        .def("get_spins", &Model::get_spins)
        .def("calc_energy", &Model::calc_energy)
        .def("random_mc", &Model::random_mc)
        .def("random_mc_meanstd", &Model::random_mc_meanstd)
        .def_readwrite("energy", &Model::energy);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
