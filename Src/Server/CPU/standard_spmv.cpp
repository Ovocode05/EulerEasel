#include<iostream>
#include<cmath>
#include<vector>
#include<random>
#include "./utils/file_parser.h"
#include "./utils/datatype.h"
#include "./utils/matrix_dim.h"
#include "./utils/create_file.h"
using namespace std;

void Csrformat(const vector<matrix_el>& matrix, int total_rows, CSR& csr, int nonz){
    /*
    .mtx data to csr format that is 3 arrays [columns_indices, row_ptr, values]
    */
   int nnz = nonz;
   cout<<"nnz"<<nnz<<endl;
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
        int sum=0;
        for(int k=csr.rptr[i]; i<csr.rptr[i+1];i++){
            int j=csr.ind[k];
            sum += csr.vals[k]*x[j];
        }

        y[i]=sum;
    }

    return y;
}

int main(){
    vector<matrix_el> matrix;
    CSR csr;

    auto [r, c, nnz] = matrix_dim();
    file_parser(matrix);
    Csrformat(matrix, r, csr, nnz);


    vector<double> x(c);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1, 1000);

    for (int i = 0; i < c; ++i) {
        x[i] = distr(gen);
    }

    //create results
    vector<double> y(r, 0);
    vector<double> y_new(r);
    y_new = SpMv_kernel(csr, x, y);

    create_outfile("/home/fakeheadset/Projects/EulerEasel/Src/Server/CPU/results", "CSR_res.txt", y_new);

} 