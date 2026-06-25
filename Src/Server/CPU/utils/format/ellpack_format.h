#include<iostream>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<string>
#include<algorithm>
#include<random>
#include<immintrin.h>
#include<omp.h>
using namespace std;

tuple<vector<vector<double>>, vector<vector<int32_t>>> ellpack_format(const vector<matrix_el>& matrix, int32_t r, int32_t c, int32_t nnz, ell& ellpack){
    ellpack.numcols=c;
    ellpack.numrows=r;

    vector<int> row_count(r,0);
    for(int32_t i=0;i<nnz;i++){
        int32_t ridx = matrix[i].row_el;
        row_count[ridx]++;
    }
    
    ellpack.max_padd = *max_element(row_count.begin(), row_count.end());

    vector<vector<double>> A(ellpack.numrows, vector<double>(ellpack.max_padd,0));
    vector<vector<int>> J(ellpack.numrows, vector<int>(ellpack.max_padd,-1));

    vector<int> row_counter(r, 0);
    for(const auto& el: matrix){
        int32_t ridx = el.row_el;
        int32_t cidx = el.col_el;
        double val = el.val_el;
        int32_t slot = row_counter[ridx];

        A[ridx][slot] =val;
        J[ridx][slot] =cidx;
        row_counter[ridx]++;
    }
    
    cout<<"A and J"<<endl;
    return {A, J};
}

vector<double> SpMv_kernel_ell(vector<double> y, vector<double> x, vector<vector<double>> A, vector<vector<int32_t>> J){
    /*
    SpMV kernel of Ellpack has shown more efficient results for GPU and parallel computations
    */
    
    int32_t numrows = A.size();
    int32_t numcols = A[0].size();

    for(int32_t i=0;i<numrows;i++){
        double sum=0;

        for(int32_t j=0;j<numcols;j++){
            double val = A[i][j];
            int32_t col = J[i][j];

            if(col==-1) break;
            sum +=val*x[col];
        }

        y[i] = sum;
    }
        
    return y;
}