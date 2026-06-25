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

tuple<vector<vector<double>>, vector<vector<int>>> ellpack_format(const vector<matrix_el>& matrix, int r, int c, int nnz, ell& ellpack){
    ellpack.numcols=c;
    ellpack.numrows=r;

    vector<int> row_count(r,0);
    for(int i=0;i<nnz;i++){
        int ridx = matrix[i].row_el;
        row_count[ridx]++;
    }
    
    ellpack.max_padd = *max_element(row_count.begin(), row_count.end());

    vector<vector<double>> A(ellpack.numrows, vector<double>(ellpack.max_padd,0));
    vector<vector<int>> J(ellpack.numrows, vector<int>(ellpack.max_padd,-1));

    vector<int> row_counter(r, 0);
    for(const auto& el: matrix){
        int ridx = el.row_el;
        int cidx = el.col_el;
        double val = el.val_el;
        int slot = row_counter[ridx];

        A[ridx][slot] =val;
        J[ridx][slot] =cidx;
        row_counter[ridx]++;
    }
    
    cout<<"A and J"<<endl;
    return {A, J};
}

vector<double> SpMv_kernel_ell(vector<double> y, vector<double> x, vector<vector<double>> A, vector<vector<int>> J){
    /*
    SpMV kernel of Ellpack has shown more efficient results for GPU and parallel computations
    */
    
    int numrows = A.size();
    int numcols = A[0].size();

    for(int i=0;i<numrows;i++){
        double sum=0;

        for(int j=0;j<numcols;j++){
            double val = A[i][j];
            int col = J[i][j];

            if(col==-1) break;
            sum +=val*x[col];
        }

        y[i] = sum;
    }
        
    return y;
}