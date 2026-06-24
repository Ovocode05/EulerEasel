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

int main(){
    vector<matrix_el> matrix;
    ell ELL;

    auto [r, c, nnz] = matrix_dim();
    file_parser(matrix);

    auto [A , J]= ellpack_format(matrix, r,c, nnz, ELL);

    // create results
    auto x = Central_Vector::generate();
    vector<double> y(r, 0);
    vector<double> y_new(r);
    y_new = SpMv_kernel_ell(y, x, A, J);

    create_outfile("/home/fakeheadset/Projects/EulerEasel/Src/Server/CPU/results", "ELL_res.txt", y_new);

}