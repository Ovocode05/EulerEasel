#include<iostream>
#include<cmath>
#include<vector>
#include<sstream>
#include<fstream>
#include<string>
#include<algorithm>
#include<random>
#include "./utils/file_parser.h"
#include "./utils/datatype.h"
#include "./utils/matrix_dim.h"
#include "./utils/create_file.h"
#include "./utils/vector_gen.h"
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

int main(){
    vector<matrix_el> matrix;
    ell ELL;

    auto [r, c, nnz] = matrix_dim();
    file_parser(matrix);
    auto [A , J]= ellpack_format(matrix, r, ELL);

    //create results
    auto x = Central_Vector::generate();
    vector<double> y(r, 0);
    vector<double> y_new(r);
    y_new = SpMv_kernel_ell(y, x, A, J);

    create_outfile("/home/fakeheadset/Projects/EulerEasel/Src/Server/CPU/results", "ELL_res.txt", y_new);

}