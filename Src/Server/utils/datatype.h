#pragma once
#include <iostream>
#include <vector>
#include <cstddef>

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define EULER_HAS_CUDA_RUNTIME 1
#else
#define EULER_HAS_CUDA_RUNTIME 0
#endif
#else
#include <cuda_runtime.h>
#define EULER_HAS_CUDA_RUNTIME 1
#endif

#if !EULER_HAS_CUDA_RUNTIME
using cudaError_t = int;

template <typename T>
inline cudaError_t cudaMalloc(T **ptr, std::size_t size)
{
    (void)ptr;
    (void)size;
    return 0;
}

template <typename T>
inline cudaError_t cudaFree(T *ptr)
{
    (void)ptr;
    return 0;
}
#endif

#include "./matrix_dim.h"
using namespace std;

struct matrix_el
{
    int32_t row_el;
    int32_t col_el;
    double val_el;
};

struct CSR
{
    int32_t num_rows;
    vector<int32_t> ind;
    vector<double> vals;
    vector<int32_t> rptr;
};

struct ell
{
    int32_t numcols;
    int32_t numrows;
    int32_t max_padd;
    vector<vector<int32_t>> col_ind;
    vector<vector<double>> val;
};

struct hybd
{
    ell el_part;
    CSR csr_part;
    vector<matrix_el> ell_entries;
    vector<matrix_el> csr_entries;
};

struct gpuCSR_Buffer
{
    int32_t *d_rptr;
    int32_t *d_ind;
    double *d_vals;
    int32_t num_rows;

    void allocate(int32_t r_size, int32_t i_size, int32_t c_size)
    {
        cudaMalloc(&d_rptr, r_size * sizeof(int32_t));
        cudaMalloc(&d_ind, c_size * sizeof(int32_t));
        cudaMalloc(&d_vals, i_size * sizeof(double));
    }

    ~gpuCSR_Buffer()
    {
        if (d_rptr)
            cudaFree(d_rptr);
        if (d_vals)
            cudaFree(d_vals);
        if (d_ind)
            cudaFree(d_ind);
    }
};