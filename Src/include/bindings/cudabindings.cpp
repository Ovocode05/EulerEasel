#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include "./../../Server/CUDA/kernel.cuh"
#include "./../../Server/CUDA/utils/cuMalloc.hpp"
#include "./../../Server/CUDA/utils/cuRegister.hpp"
#include "./../../Server/CUDA/utils/cuStream.hpp"
#include "./../../Server/CUDA/utils/flat.hpp"
#include "./../../Server/utils/datatype.h"

using namespace std;
namespace py = pybind11;

#ifndef EULER_PYBIND_MODULE_NAME
#define EULER_PYBIND_MODULE_NAME CUDAruntime
#endif

template <typename T>
void bind_device_buffer_type(
    py::module_ &m,
    const std::string &python_class_name)
{
    py::class_<deviceBuffer<T>>(m, python_class_name.c_str())

        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("size"))

        .def("getsize",
             &deviceBuffer<T>::size)

        .def_property_readonly(
            "ptr",
            [](deviceBuffer<T> &self)
            {
                return reinterpret_cast<uintptr_t>(self.data());
            })

        .def(
            "h2d",
            [](deviceBuffer<T> &self,
               py::array_t<T,
                           py::array::c_style |
                               py::array::forcecast>
                   array)
            {
                auto info = array.request();

                size_t count =
                    static_cast<size_t>(info.size);

                if (count > self.size())
                    throw std::runtime_error(
                        "Input array exceeds device buffer size");

                self.host_to_device(
                    static_cast<const T *>(info.ptr),
                    count);

                cudaStreamSynchronize(0);
                CUDA_CHECK();
            },
            py::arg("array"))

        .def(
            "d2h",
            [](deviceBuffer<T> &self)
            {
                py::array_t<T> output(self.size());

                auto info = output.request();

                self.device_to_host(
                    static_cast<T *>(info.ptr),
                    self.size());

                cudaStreamSynchronize(0);
                CUDA_CHECK();

                return output;
            });
}
PYBIND11_MODULE(EULER_PYBIND_MODULE_NAME, cudarn)
{
    cudarn.doc() = "this is the cuda framework of the library that allows the user to compute matrix vector multiplication";

    // cuda classes
    py::class_<gpuCSR_Buffer>(cudarn, "cudaBuffer")
        .def(py::init<>())
        .def("allocate",
             &gpuCSR_Buffer::allocate,
             py::arg("r_size"),
             py::arg("i_size"),
             py::arg("c_size"))
        .def_readwrite("nrows", &gpuCSR_Buffer::num_rows)
        .def_property_readonly("d_rptr", [](gpuCSR_Buffer &self)
                               { return reinterpret_cast<uintptr_t>(self.d_rptr); })
        .def_property_readonly("d_ind", [](gpuCSR_Buffer &self)
                               { return reinterpret_cast<uintptr_t>(self.d_ind); })
        .def_property_readonly("d_vals", [](gpuCSR_Buffer &self)
                               { return reinterpret_cast<uintptr_t>(self.d_vals); });

    // malloc
    bind_device_buffer_type<double>(cudarn, "deviceBufferDouble");
    bind_device_buffer_type<int32_t>(cudarn, "deviceBufferInt");

    // unregister
    py::class_<cudaRegister<double>>(
        cudarn,
        "cudaRegisterDouble")
        .def(
            py::init<size_t, uintptr_t>(),
            py::arg("size"),
            py::arg("ptr"))

        .def(
            "__enter__",
            [](cudaRegister<double> &self)
                -> cudaRegister<double> &
            {
                return self;
            },
            py::return_value_policy::reference_internal)

        .def(
            "__exit__",
            [](cudaRegister<double> &self,
               py::object,
               py::object,
               py::object) {});

    // streamm
    py::class_<CudaStream>(cudarn, "cudaStream")
        .def(py::init<bool>(), py::arg("non_blocking"))
        .def_property_readonly(
            "ptr",
            [](const CudaStream &self)
            {
                return reinterpret_cast<uintptr_t>(self.get());
            });

    // flat and error
    cudarn.def("flatten", &flatten, "flat the 2dim matrix into a 1dim vector");

    // spmv gpu
    cudarn.def("cuda_csr", [](int32_t threads, int32_t blocks, deviceBuffer<int32_t> &d_rptr, deviceBuffer<int32_t> &d_ind, deviceBuffer<double> &d_vals, int32_t numrows, deviceBuffer<double> &d_x, deviceBuffer<double> &d_y)
               { return spmv_csr(threads,
                                 blocks,
                                 d_rptr.data(),
                                 d_ind.data(),
                                 d_vals.data(),
                                 numrows,
                                 d_x.data(),
                                 d_y.data()); }, py::arg("threads"), py::arg("blocks"), py::arg("d_rptr"), py::arg("d_ind"), py::arg("d_vals"), py::arg("numrows"), py::arg("d_x"), py::arg("d_y"));

    cudarn.def("cuda_ell", [](int32_t threads, int32_t blocks, deviceBuffer<double> &d_a, deviceBuffer<int32_t> &d_j, int32_t rows, int32_t cols, deviceBuffer<double> &d_x, deviceBuffer<double> &d_y)
               { 
                 Matrix<double> A{
                d_a.data(),
                rows,
                cols
            };

            Matrix<int32_t> J{
                d_j.data(),
                rows,
                cols
            };
            
            return spmv_ell(
                     threads,
                     blocks,
                     A, J, d_x.data(), d_y.data()); }, py::arg("threads"), py::arg("blocks"), py::arg("A"), py::arg("J"), py::arg("rows"), py::arg("cols"), py::arg("d_x"), py::arg("d_y"));

    cudarn.def("cuda_hyb", [](int32_t threads, int32_t blocks, deviceBuffer<int32_t> &d_rptr, deviceBuffer<int32_t> &d_ind, deviceBuffer<double> &d_vals, deviceBuffer<double> &d_a, deviceBuffer<int> &d_j, int32_t ell_rows, int32_t csr_rows, int32_t cols, deviceBuffer<double> &d_x, deviceBuffer<double> &d_y, CudaStream &stream1, CudaStream &stream2)
               {
                    Matrix<double> A{
                    d_a.data(),
                    ell_rows,
                    cols
                };

                Matrix<int32_t> J{
                    d_j.data(),
                    ell_rows,
                    cols
                };

                return spmv_hybd(
                            threads,
                            blocks,
                            d_rptr.data(),
                            d_ind.data(),
                            d_vals.data(),
                            A, J,
                            ell_rows,
                            csr_rows,
                            d_x.data(),
                            d_y.data(),
                            stream1.get(),
                            stream2.get()); }, py::arg("threads"), py::arg("blocks"), py::arg("d_rptr"), py::arg("d_ind"), py::arg("d_vals"), py::arg("d_a"), py::arg("d_j"), py::arg("ell_rows"), py::arg("csr_rows"), py::arg("cols"), py::arg("d_x"), py::arg("d_y"), py::arg("stream1"), py::arg("stream2"));
}
