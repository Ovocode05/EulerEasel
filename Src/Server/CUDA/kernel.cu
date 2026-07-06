#include <iostream>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <vector>
#include "../utils/matrix_dim.h"
#include "../utils/datatype.h"
#include "../utils/file_parser.h"
#include "./../utils/vector_gen.h"
#include "../utils/create_file.h"
#include "../utils/format/csr_format.h"
#include "../utils/format/ellpack_format.h"
using namespace std;

__global__ void spmv_kernel_C(
    const int32_t *__restrict__ d_rptr,
    const int32_t *__restrict__ d_ind,
    const double *__restrict__ d_vals,
    const int32_t nrows,
    const double *__restrict__ x,
    double *y)
{
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < nrows; i += stride)
    {
        double sum = 0.0;

        int32_t r_start = d_rptr[i];
        int32_t r_end = d_rptr[i + 1];
        for (int32_t j = r_start; j < r_end; ++j)
            sum += x[j] * d_vals[d_ind[j]];

        y[i] = sum;
    }
}

int main()
{
    vector<matrix_el> Mat;
    file_parser("/home/fakeheadset/Projects/EulerEasel/Data/bcsstk18.mtx", Mat);

    // creating the threads
    int32_t threads = 256;
    int32_t blocks = cuda::ceil_div(Mat.size(), threads);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto [r, c, nnz] = matrix_dim("/home/fakeheadset/Projects/EulerEasel/Data/bcsstk18.mtx");
    CSR csr;
    Csrformat(Mat, r, c, nnz, csr);
    vector<double> y_cpu(r, 0.0);

    gpuCSR_Buffer d_csr;
    d_csr.allocate(csr.rptr.size(), csr.vals.size(), csr.ind.size());
    d_csr.num_rows = csr.num_rows;
    cudaMemcpyAsync(d_csr.d_rptr, csr.rptr.data(), sizeof(int32_t) * csr.rptr.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_csr.d_vals, csr.vals.data(), sizeof(double) * csr.vals.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_csr.d_ind, csr.ind.data(), sizeof(int32_t) * csr.ind.size(), cudaMemcpyHostToDevice, stream);

    // y is a raw pointer, hence no member call like .data()
    double *y = nullptr;
    cudaMalloc(&y, sizeof(double) * csr.num_rows);
    cudaMemset(y, 0.0, sizeof(double) * csr.num_rows);

    vector<double> x = Central_Vector::generate(r, c, nnz);
    double *d_x = nullptr;
    cudaMalloc(&d_x, sizeof(int32_t) * x.size());
    // pin the CPU vector memory pages
    cudaHostRegister(x.data(), sizeof(int32_t) * x.size(), cudaHostRegisterDefault);
    cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // launch the kernel
    spmv_kernel_C<<<blocks, threads, 0, stream>>>(d_csr.d_rptr, d_csr.d_ind, d_csr.d_vals, d_csr.num_rows, d_x, y);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }
    cudaMemcpy(y_cpu.data(), y, sizeof(double) * r, cudaMemcpyDeviceToHost);
    create_outfile("/home/fakeheadset/Projects/EulerEasel/Src/Server/CUDA/results", "cuda_res.txt", y_cpu);

    cudaFree(y);
    cudaFree(d_x);
    cudaStreamDestroy(stream);

    return 0;
}