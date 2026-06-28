#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <tuple>
#include<vector>
using namespace std;

#include "./hybrid_ell-csr.cpp"
PYBIND11_MAKE_OPAQUE(vector<matrix_el>); 
namespace py = pybind11;

PYBIND11_MODULE(eulereasel, m){
    m.doc()= "Python bindings for EulerEasel High-Performance CPU Kernels";

    py::module_ m_data = m.def_submodule("datatype", "Matrix storage structures");
    py::bind_vector<vector<matrix_el>>(m_data, "mat_vec");

    py::class_<matrix_el>(m_data, "MATRIXEL")
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

// #343fff functions
    py::module_ m_funcs = m.def_submodule("functions", "High-performance processing operations");
    m_funcs.def("file_parser", &file_parser, "read the raw file and convert to vector of matrix element",
          py::arg("filename"), py::arg("matrix"));

    m_funcs.def("matdim", &matrix_dim, "Find the matrix dimensions", py::arg("filename"));

    py::class_<Central_Vector>(m_funcs, "CentralVector")
          .def(py::init<>())
          .def_static("generate", &Central_Vector::generate, 
            "Generate a reproducible dense input vector matching matrix constraints",
            py::arg("r"), py::arg("c"), py::arg("nnz"));
  
// #343fff csr submodule
    py::module_ mfunc_t = m_funcs.def_submodule("csr", "Only CSR");

    mfunc_t.def("csrformat", &Csrformat, "Convert .mtx matrix array into CSR format",
          py::arg("matrix"), py::arg("r"), py::arg("c"), py::arg("nnz"), py::arg("csr"));

    mfunc_t.def("sparse", &SpMv_kernel, "Standard Sparse Matrix-Vector Multiplication",
          py::arg("csr"), py::arg("x"));

    mfunc_t.def("sparseAVX_x4", &SpMV_kernel_AVX, "OMP + AVX Vectorized SpMV Multiplication",
          py::arg("csr"), py::arg("x"));
      
//#343fff ell submodule
    py::module_ mfunc_t2 = m_funcs.def_submodule("ell", "Only ELL");    

    mfunc_t2.def("ellformat", &ellpack_format, "Convert .mtx matrix array into Ellpack format",
          py::arg("matrix"), py::arg("r"), py::arg("c"), py::arg("nnz"), py::arg("ellpack"));

    mfunc_t2.def("sparse", &SpMv_kernel_ell, "Standard Sparse Matrix-Vector Multiplication",
          py::arg("x"), py::arg("A"), py::arg("J"));

    mfunc_t2.def("sparseAVX_x4", &ell_spMV_AVX, "OMP + AVX Vectorized SpMV Multiplication",
          py::arg("x"), py::arg("A"), py::arg("J"));

    mfunc_t2.def("sparseAVX_x16", &ell_pack_AVX_vertical, "OMP + AVX Vectorized SpMV Multiplication",
          py::arg("x"), py::arg("A"), py::arg("J"));

// #343fff hybrid sumodule
    py::module_ mfunc_t3 = m_funcs.def_submodule("hybrid", "Only hybrid");

    mfunc_t3.def("hybridformat", &hybrid_format, "Split matrix into hybrid format",
          py::arg("hybrid"), py::arg("matrix"), py::arg("r"), py::arg("c"), py::arg("nnz"));

    mfunc_t3.def("sparse", &SpMv_kernel_hybrid, "Hybrid SpMV multiplication execution",
          py::arg("hybrid"), py::arg("x"), py::arg("A"), py::arg("J"), py::arg("r"));



}



