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
    int32_t total_rows = y.size();
    
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
    size 256 bits AVX registers
    */

    int32_t num_rows = csr.num_rows; 
    int32_t i=0;
    for(; i<num_rows-4;i+=4){
        //initialize the accumulator
        

    }

    //creating the register that accepts the data
    //loading the data into the register
    //fuse addtion : y += x*A
    return y;
}