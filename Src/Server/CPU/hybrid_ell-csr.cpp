#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<cmath>
#include "./utils/datatype.h"
#include "./utils/format/csr_format.h"
#include "./utils/format/ellpack_format.h"
#include "./utils/vector_gen.h"
#include "./utils/create_file.h"
#include "./utils/file_parser.h"
using namespace std;

tuple<vector<vector<double>>, vector<vector<int>>>hybrid_format(hybd& hybrid, vector<matrix_el>& matrix, int r, int c, int nnz){

    vector<int> row_counter(r, 0);
    for(const auto& el : matrix){
        row_counter[el.row_el]++;
    }
    
    vector<int> sorted = row_counter;
    sort(sorted.begin(), sorted.end());
    int threshold = sorted[0];
    int max_jump = 0;
    for(size_t i=1;i<sorted.size();i++){
        int jump = sorted[i] - sorted[i-1];
        if(jump > max_jump){
            max_jump = jump;
            threshold = sorted[i-1];
        }
    }

    for(const auto& el: matrix){
        int row = el.row_el;
        if(row_counter[row]<=threshold){
            //apply csr
            hybrid.ell_entries.emplace_back(el);
        }else{
            // apply ell
            hybrid.csr_entries.emplace_back(el);
        }
    }

    //measure r,c,nnz for each part of the matrix
    int nnz_csr = hybrid.csr_entries.size();
    int nnz_ell = hybrid.ell_entries.size();

    Csrformat(hybrid.csr_entries, r,c,nnz_csr, hybrid.csr_part);
    auto [A,J] = ellpack_format(hybrid.ell_entries, r, c,nnz_ell, hybrid.el_part);

    return {A,J};
}

vector<double> SpMv_kernel_hybrid(hybd& hybrid, vector<double> y, const vector<double>& x, vector<vector<double>>& A, vector<vector<int>>& J, int r){
    auto y_csr = SpMv_kernel(hybrid.csr_part, x, y);
    auto y_ell = SpMv_kernel_ell(y,x, A,J); 
    vector<double> y_new(r,0);

    for(int i = 0; i < r; i++){
        y_new[i] = y_csr[i] + y_ell[i];
    }

    return y_new;

}

int main(){
    hybd hybrid;
    vector<matrix_el> matrix;
    file_parser(matrix);
    auto [r, c, nnz] = matrix_dim();
    auto [A, J] = hybrid_format(hybrid,matrix, r,c,nnz);

    auto x = Central_Vector::generate();
    vector<double> y(r,0);
    vector<double> y_new(r);

    y_new = SpMv_kernel_hybrid(hybrid, y, x, A, J ,r);
    create_outfile("/home/fakeheadset/Projects/EulerEasel/Src/Server/CPU/results", "Hybrid_res.txt", y_new);
    
    return 0;
}