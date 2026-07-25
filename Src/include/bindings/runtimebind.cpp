#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "./../Strategy/strategy.hpp"
#include "./../Matrix/extractor.hpp"
#include "./../../Server/CPU/hybrid_ell-csr.h"
#include "./../../Server/utils/format/csr_format.h"
#include "./../../Server/utils/format/ellpack_format.h"

using namespace std;
namespace py = pybind11;

#ifndef EULER_PYBIND_MODULE_NAME
#define EULER_PYBIND_MODULE_NAME runtime
#endif

PYBIND11_MODULE(EULER_PYBIND_MODULE_NAME, run)
{
    run.doc() = "i will come to this later";

    // spmv cpu
    run.def("proc_csr", &SpMv_kernel, "a csr spmv kernel that performs the computation completely in CPU");
    run.def("proc_csrx4", &SpMV_kernel_AVX, "a csr spmv kernel that performs the computation completely on CPU using AVX/ openMP features");
    run.def("proc_ell", &SpMv_kernel_ell, "a ellpack spmv kernel that performs the computation completely on CPU");
    run.def("proc_ellx4", &ell_spMV_AVX, "a ellpack spmv kernel that performs the computation completely on CPU using AVX/openMP features");
    run.def("proc_ell_x16", &ell_pack_AVX_vertical, "a ellpack spmv kernel that performs the computation completely on CPU using AVX/openMP features");
    run.def("proc_hyb", &SpMv_kernel_hybrid, "a hybrid spmv kernel that performs the computation completely on CPU using AVX/openMP features");
    run.def("proc_hyb_avx4", &SpMv_kernel_hybrid_x4, "a hybrid spmv kernel that performs the computation completely on CPU using AVX/openMP features");
    run.def("proc_hyb_avx16", &SpMv_kernel_hybrid_16x, "a hybrid spmv kernel that performs the computation completely on CPU using AVX/openMP features");
}