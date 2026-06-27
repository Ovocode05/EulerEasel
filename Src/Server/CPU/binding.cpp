#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>

#include "./hybrid_ell-csr.cpp"
namespace py = pybind11;

PYBIND11_MODULE(eulereasel, m){
    m.doc()= "Python bindings for EulerEasel High-Performance CPU Kernels";

    py::module_ m_data = m.def_submodule("datatype", "Matrix storage structures");
    py::class_<matrix_el>(m_data, "MatrixEl")
        .def(py::init<>())
        .def_readwrite("row_el", &matrix_el::row_el)
        .def_readwrite("col_el", &matrix_el::col_el)
        .def_readwrite("val_el", &matrix_el::val_el);

    py::class_<CSR>(m_data, "CSR")
        .def(py::init<>())
        .def_readwrite("rptr", &CSR::rptr)
        .def_readwrite("ind", &CSR::ind)
        .def_readwrite("vals", &CSR::vals)
        .def_readwrite("num_rows", &CSR::num_rows);

    py::class_<ell>(m_data, "ELL")
        .def(py::init<>())
        .def_readwrite("numcols", &ell::numcols)
        .def_readwrite("numrows", &ell::numrows)
        .def_readwrite("max_padd", &ell::max_padd)
        .def_readwrite("val", &ell::val)
        .def_readwrite("col_ind", &ell::col_ind);
        
    py::class_<hybd>(m_data, "HYBD")
        .def(py::init<>())
        .def_readwrite("el_part", &hybd::el_part)
        .def_readwrite("csr_part", &hybd::csr_part)
        .def_readwrite("ell_entries", &hybd::ell_entries)
        .def_readwrite("csr_entries", &hybd::csr_entries);


    py::module_ m_funcs = m.def_submodule("functions", "High-performance processing operations");
    m_funcs.def("Csrformat", &Csrformat, "Convert .mtx matrix array into CSR format",
          py::arg("matrix"), py::arg("r"), py::arg("c"), py::arg("nnz"), py::arg("csr"));

    m_funcs.def("SpMv_kernel", &SpMv_kernel, "Standard Sparse Matrix-Vector Multiplication",
          py::arg("csr"), py::arg("x"));

    m_funcs.def("SpMV_kernel_AVX", &SpMV_kernel_AVX, "OMP + AVX Vectorized SpMV Multiplication",
          py::arg("csr"), py::arg("x"));

    m_funcs.def("ellpack_format", &ellpack_format, "Convert .mtx matrix array into Ellpack format",
          py::arg("matrix"), py::arg("r"), py::arg("c"), py::arg("nnz"), py::arg("ellpack"));

    m_funcs.def("SpMv_kernel_ell", &SpMv_kernel_ell, "Standard Sparse Matrix-Vector Multiplication",
          py::arg("x"), py::arg("A"), py::arg("J"));

    m_funcs.def("ell_spMV_AVX", &ell_spMV_AVX, "OMP + AVX Vectorized SpMV Multiplication",
          py::arg("x"), py::arg("A"), py::arg("J"));

    m_funcs.def("ell_spMV_AVX_vertical", &ell_pack_AVX_vertical, "OMP + AVX Vectorized SpMV Multiplication",
          py::arg("x"), py::arg("A"), py::arg("J"));

    m_funcs.def("hybrid_format", &hybrid_format, "Split matrix into hybrid format",
          py::arg("hybrid"), py::arg("matrix"), py::arg("r"), py::arg("c"), py::arg("nnz"));

    m_funcs.def("SpMv_kernel_hybrid", &SpMv_kernel_hybrid, "Hybrid SpMV multiplication execution",
          py::arg("hybrid"), py::arg("x"), py::arg("A"), py::arg("J"), py::arg("r"));
}



