#include<iostream>
#include<cmath>
#include<vector>
#include "./utils/file_parser.h"
#include "./utils/datatype.h"
using namespace std;

void Csrformat(const vector<matrix_el>& matrix, int total_rows, CSR& csr){
    /*
    .mtx data to csr format that is 3 arrays [columns_indices, row_ptr, values]
    */
   int nnz = matrix.size();
   csr.num_rows = nnz;
   csr.ind.resize(nnz);
   csr.vals.resize(nnz);
   csr.rptr.assign(total_rows+1,0);

   for(int i=0;i<nnz;i++){
        csr.vals[i] = matrix[i].val_el;
        csr.ind[i] = matrix[i].col_el;
        int r = matrix[i].row_el;
        csr.rptr[r+1]++; 
   }

   for(int i=1;i<total_rows;i++){
        csr.rptr[i] +=csr.rptr[i-1];
   }
   
   cout<<"CSR built!"<<endl;
}

vector<double> SpMv_kernel(CSR& csr, vector<double> x, vector<double> y){
    /*
    SpMv_kernel forms the basic sparse vector multiplication function that 
    */
    int total_rows = y.size();
    for(int i=0;i<total_rows;i++){
        for(int k=csr.rptr[i]; i<csr.rptr[i+1];i++){
            int j=csr.ind[k];
            y[i] += csr.vals[k]*x[j];
        }
    }

    return y;
}

