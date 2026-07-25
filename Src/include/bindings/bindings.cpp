#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./../Strategy/strategy.hpp"
#include "./../Matrix/extractor.hpp"
#include "./../../Server/CPU/hybrid_ell-csr.h"
#include "./../../Server/utils/create_file.h"
#include "./../../Server/utils/format/ellpack_format.h"
using namespace std;
namespace py = pybind11;

#ifndef EULER_PYBIND_MODULE_NAME
#define EULER_PYBIND_MODULE_NAME matrix_extractor
#endif

PYBIND11_MODULE(EULER_PYBIND_MODULE_NAME, mat)
{
    py::class_<matrix_el>(mat, "MatrixEl")
        .def(py::init<>())
        .def_readwrite("row_el", &matrix_el::row_el)
        .def_readwrite("col_el", &matrix_el::col_el)
        .def_readwrite("val_el", &matrix_el::val_el);

    py::class_<MatrixFeatures>(mat, "MatrixFeatures");

    py::class_<CSR>(mat, "CSR")
        .def(py::init<>())
        .def_readwrite("num_rows", &CSR::num_rows)
        .def_readwrite("ind", &CSR::ind)
        .def_readwrite("vals", &CSR::vals)
        .def_readwrite("rptr", &CSR::rptr);

    py::class_<ell>(mat, "ELL")
        .def(py::init<>())
        .def_readwrite("numcols", &ell::numcols)
        .def_readwrite("numrows", &ell::numrows)
        .def_readwrite("maxpad", &ell::max_padd)
        .def_readwrite("A", &ell::val)
        .def_readwrite("J", &ell::col_ind);

    py::class_<hybd>(mat, "HYB")
        .def(py::init<>())
        .def_readwrite("csr", &hybd::csr_entries)
        .def_readwrite("ell", &hybd::ell_entries)
        .def_readwrite("vec_c", &hybd::csr_part)
        .def_readwrite("vec_e", &hybd::el_part);

    mat.def("file_paser", &file_parser, "Parse a matrix from a file");
    mat.def("csrformat", &Csrformat, "Convert a dense matrix to CSR format");
    mat.def("ellformat", &ellpack_format, "Convert a dense matrix to ELL format");
    mat.def("hybd_format", &hybrid_format, "Convert a dense matrix to hybrid format");
    mat.def("mat_dim", &matrix_dim, "Get the dimensions of a matrix from a file");
    mat.def("results", &create_outfile, "results");

    py::class_<CsrMatrix<double>>(mat, "CsrMatrix")
        .def(py::init<CSR &, int32_t>())
        .def("get_nrow", &CsrMatrix<double>::get_nrow)
        .def("get_ncol", &CsrMatrix<double>::get_ncol)
        .def("get_nnz", &CsrMatrix<double>::get_nnz)
        .def("get_rptr", &CsrMatrix<double>::get_rptr)
        .def("get_colidx", &CsrMatrix<double>::get_colidx)
        .def("get_vals", &CsrMatrix<double>::get_vals);

    py::class_<MatrixExtractor<double>>(mat, "MatrixExtractor")
        .def(py::init<CsrMatrix<double> &>())
        .def("extract_all", &MatrixExtractor<double>::extract_all)
        .def("to_flat_vector", &MatrixExtractor<double>::to_flat_vector);

    py::class_<StrategyRegister>(mat, "StrategyRegister")
        .def(py::init<>())
        .def("get_strategies", &StrategyRegister::get_strategies, "Get available strategies based on hardware context")
        .def("generate", &StrategyRegister::generate, "Generate available strategies based on hardware context")
        .def("create", &StrategyRegister::create, "Create a list of all possible strategies")
        .def("get_name", &StrategyRegister::get_strategy_names, "Get the names");

    py::class_<Hardware_filter>(mat, "Hardware_filter")
        .def(py::init<>())
        .def("apply", &Hardware_filter::apply, "Apply hardware filter to strategies based on hardware context");

    py::class_<HardwareContext>(mat, "HardwareContext")
        .def(py::init<>())
        .def_readwrite("cpu", &HardwareContext::cpu)
        .def_readwrite("gpu", &HardwareContext::gpu)
        .def_readwrite("has_gpu", &HardwareContext::has_gpu)
        .def_readwrite("has_openMP", &HardwareContext::has_openMP)
        .def_readwrite("has_AVX", &HardwareContext::has_AVX)
        .def_readwrite("gpu_cc", &HardwareContext::gpu_cc)
        .def_readwrite("h2d_bandwidth_gbs", &HardwareContext::h2d_bandwidth_gbs)
        .def_readwrite("d2h_bandwidth_gbs", &HardwareContext::d2h_bandwidth_gbs);

    py::class_<strategy>(mat, "strategy")
        .def(py::init<>())
        .def_readwrite("device", &strategy::device)
        .def_readwrite("format", &strategy::format)
        .def_readwrite("arch", &strategy::arch)
        .def_readwrite("kernel", &strategy::kernel)
        .def_readwrite("is_available", &strategy::is_available);
}
