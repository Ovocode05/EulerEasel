#include<iostream>
#include<cmath>
#include<vector>
#include<random>
#include<immintrin.h>
#include<omp.h>
using namespace std;

void Csrformat(const vector<matrix_el>& matrix, int32_t r, int32_t c, int32_t nnz,CSR& csr){
    /*
    .mtx data to csr format that is 3 arrays [columns_indices, row_ptr, values]
    */
   csr.num_rows = r;
   csr.ind.resize(nnz);
   csr.vals.resize(nnz);
   csr.rptr.assign(r+1,0);

   for(int32_t i=0;i<nnz;i++){
        csr.vals[i] = matrix[i].val_el;
        csr.ind[i] = matrix[i].col_el;
        int32_t r = matrix[i].row_el;
        csr.rptr[r+1]++; 
   }

   for(int32_t i=1;i<=r;i++){
        csr.rptr[i] +=csr.rptr[i-1];
   }
   
   cout<<"CSR built!"<<endl;
}

vector<double> SpMv_kernel(CSR& csr, vector<double> x, vector<double> y){
    /*
    SpMv_kernel forms the basic sparse vector multiplication function that 
    */
    int32_t total_rows = csr.num_rows;
    
    for(int32_t i=0;i<total_rows;i++){
        double sum=0;
        for(int32_t k=csr.rptr[i]; k<csr.rptr[i+1];k++){
            int32_t j=csr.ind[k];
            sum += csr.vals[k]* x[j];
        }

        y[i]=sum;
    }

    return y;
}

vector<double> SpMV_kernel_AVX(CSR& csr, vector<double> x, vector<double> y){
    /*
    The AVX intrinsic implemented for the SpMV kernel for csr format 
    int32_t bits variable storage
    size 256 bits double precision floating point YMM registers

    steps: initializing 4*64bits precision floats with 0.0
    creating size width 32bits signed integer variables for iterating over row ptr
    loading 4 contiguous double vals into __m256d and similarly for __m128i
    gather x.data scattered in memory - 4 blocks from RAM  
    fuse addition
    */

    int32_t num_rows = csr.num_rows; 

    for(int32_t i=0; i<num_rows;i++){

        __m256d vec_sum = _mm256_setzero_pd();
        int32_t k = csr.rptr[i];
        int32_t rowend = csr.rptr[i+1];
        
        for(; k<rowend-4;k+=4){

            __m256d vals = _mm256_loadu_pd(&csr.vals[k]);
            __m128i ind = _mm_loadu_si128((const __m128i*)&csr.ind[k]);

            __m256d vec_x = _mm256_i32gather_pd(x.data(), ind, 8);

            vec_sum = _mm256_fmadd_pd(vals, vec_x, vec_sum);
        }

        double row_sum_array[4];
        _mm256_store_pd(row_sum_array, vec_sum);
        double final_sum_array = row_sum_array[0] + row_sum_array[1] + row_sum_array[2] + row_sum_array[3];
        
        for(; k<rowend;k++) final_sum_array += csr.vals[k]*x[csr.ind[k]];

        y[i] = final_sum_array;
    }

   return y;
}