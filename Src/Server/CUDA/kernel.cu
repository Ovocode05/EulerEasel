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
#include "./../CPU/hybrid_ell-csr.h"
using namespace std;

template <typename T>

struct Matrix
{
    T *data;
    int32_t rows;
    int32_t cols;

    __host__ __device__ T &operator()(int32_t row, int32_t col)
    {
        return data[(col * rows) + row];
    }

    __host__ __device__ const T &operator()(int32_t row, int32_t col) const
    {
        return data[(col * rows) + row];
    }
};

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

        atomicAdd(&y[i], sum);
    }
}

__global__ void spmv_kernel_E(
    const Matrix<double> A,
    const Matrix<int32_t> J,
    const double *__restrict__ x,
    double *y)
{
    // avoid shared memory and tiling: no reuse of x values
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = idx; i < J.rows; i += stride)
    {
        double sum = 0.0;
        for (int32_t j = 0; j < J.cols; ++j)
        {
            // coalesced access : columns major form
            int32_t col = J(i, j);
            if (col < 0)
                continue;
            double val = A(i, j);
            sum += val * x[col];
        }

        atomicAdd(&y[i], sum);
    }
}

void Hybrid(const int32_t *__restrict__ d_rptr,
            const int32_t *__restrict__ d_ind,
            const double *__restrict__ d_vals,
            int32_t nrows_csr,
            double *x,
            Matrix<double> A,
            Matrix<int32_t> J,
            int32_t nrows_ell,
            double *y,
            int32_t threads,
            cudaStream_t stream1,
            cudaStream_t stream2)
{
    int32_t blocks_csr = (nrows_csr + threads - 1) / threads;
    int32_t blocks_ell = (nrows_ell + threads - 1) / threads;
    spmv_kernel_C<<<blocks_ell, threads, 0, stream1>>>(d_rptr, d_ind, d_vals, nrows_csr, x, y);
    spmv_kernel_E<<<blocks_csr, threads, 0, stream2>>>(A, J, x, y);
}

int main()
{
    vector<matrix_el> Mat;
    Mat = file_parser("/home/fakeheadset/Projects/EulerEasel/Data/bcsstk18.mtx");

    // #343fff creating the threads
    int32_t threads = 256;
    int32_t blocks = cuda::ceil_div(Mat.size(), threads);
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

    // #343fff
    auto [r, c, nnz] = matrix_dim("/home/fakeheadset/Projects/EulerEasel/Data/bcsstk18.mtx");
    hybd hybrid;
    vector<double> y_cpu(r, 0.0);

    auto [A, J, csr] = hybrid_format(hybrid, Mat, r, c, nnz);

    gpuCSR_Buffer d_csr;
    d_csr.num_rows = csr.num_rows;
    d_csr.allocate(csr.rptr.size(), csr.vals.size(), csr.ind.size());
    Matrix<double> A_gpu;
    Matrix<int32_t> J_gpu;
    A_gpu.rows = A.size();
    A_gpu.cols = A[0].size();
    J_gpu.rows = J.size();
    J_gpu.cols = J[0].size();
    cudaMallocAsync(&A_gpu.data, sizeof(double) * A.size() * A[0].size(), stream2);
    cudaMallocAsync(&J_gpu.data, sizeof(int32_t) * J.size() * J[0].size(), stream2);
    vector<double> A_flat(A.size() * A[0].size());
    vector<int32_t> J_flat(J.size() * J[0].size());

    for (size_t i = 0; i < A.size(); ++i)
    {
        for (size_t j = 0; j < A[0].size(); ++j)
        {
            A_flat[j * r + i] = A[i][j];
            J_flat[j * r + i] = J[i][j];
        }
    }

    cudaMemcpyAsync(A_gpu.data, A_flat.data(), sizeof(double) * A.size() * A[0].size(), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(J_gpu.data, J_flat.data(), sizeof(int32_t) * A.size() * A[0].size(), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_csr.d_rptr, csr.rptr.data(), sizeof(int32_t) * csr.rptr.size(), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_csr.d_ind, csr.ind.data(), sizeof(int32_t) * csr.ind.size(), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_csr.d_vals, csr.vals.data(), sizeof(double) * csr.vals.size(), cudaMemcpyHostToDevice, stream1);

    // #343fff y is a raw pointer, hence no member call like .data()
    double *y = nullptr;
    cudaMalloc(&y, sizeof(double) * r);
    cudaMemset(y, 0.0, sizeof(double) * r);

    vector<double> x = Central_Vector::generate(r, c, nnz);
    double *d_x = nullptr;
    cudaMalloc(&d_x, sizeof(double) * x.size());
    // pin the CPU vector memory pages
    cudaHostRegister(x.data(), sizeof(double) * x.size(), cudaHostRegisterDefault);
    cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice, stream1);

    // #343fff launch the kernel
    Hybrid(d_csr.d_rptr, d_csr.d_ind, d_csr.d_vals, d_csr.num_rows, d_x, A_gpu, J_gpu, A_gpu.rows, y, threads, stream1, stream2);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }
    cudaMemcpy(y_cpu.data(), y, sizeof(double) * r, cudaMemcpyDeviceToHost);
    create_outfile("/home/fakeheadset/Projects/EulerEasel/Src/Server/CUDA/results", "cuda_res_hybrid.txt", y_cpu);

    cudaFree(y);
    cudaFree(d_x);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}