#include<iostream>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<string>
#include<algorithm>
#include "./utils/file_parser.h"
#include "./utils/datatype.h"
#include "./utils/matrix_dim.h"
using namespace std;

tuple<vector<vector<double>>, vector<vector<int>>> ellpack_format(const vector<matrix_el>& matrix,  int total_rows, ell& ellpack){
    auto [numrows, numcols, nnz] = matrix_dim(); 
    ellpack.numcols=numcols;
    ellpack.numrows=numrows;

    vector<int> row_ptr(total_rows+1,0);
    for(int i=0;i<nnz;i++){
        int r = matrix[i].row_el;
        row_ptr[r+1]++;
    }
    
    sort(row_ptr.begin(), row_ptr.end());
    ellpack.max_padd = row_ptr.back();

    vector<vector<double>> A(ellpack.numrows, vector<double>(ellpack.max_padd,0));
    vector<vector<int>> J(ellpack.numrows, vector<int>(ellpack.max_padd,0));

    vector<int> row_counter(numrows, 0);
    for(const auto& el: matrix){
        int r = el.row_el;
        int c =el.col_el;
        double val = el.val_el;
        int slot = row_counter[r];

        A[r][slot] =val;
        J[r][slot] =c;
        row_counter[r]++;
    }

    //columns order 
    vector<double> flat_A;
    vector<int> flat_J;
    for(size_t col=0;col<numcols;col++){
        for(size_t row=0;row<numrows;row++){
            flat_A.emplace_back(A[row][col]);
            flat_J.emplace_back(J[row][col]);
        }
    }
    
    return {A, J};
}

// void ellpack_row_format(const vector<matrix_el>& matrix,  int total_rows, ell& ellpack){
//     return;
// }

vector<double> SpMv_kernel_ell(vector<double> y, vector<int> x, vector<vector<double>> A, vector<vector<double>> J){
    /*
    SpMV kernel of Ellpack has shown more efficient results for GPU and parallel computations
    */
    
    int numrows = A.size();
    int numcols = A[0].size();
    for(int i=0;i<numrows;i++){
        int sum=0;

        for(int j=0;j<numcols;j++){
            int val = A[i][j];
            int col = J[i][j];
            sum +=val*x[col];
        }

        y[i] = sum;
    }
        
    return y;
}