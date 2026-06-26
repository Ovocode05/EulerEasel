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

vector<double> ell_spMV_AVX(vector<double> y, vector<double> x, vector<vector<double>> A, vector<vector<int32_t>> J){
    /*
    The AVX intrinsic implemented for the SpMV kernel for csr format 
    int32_t bits variable storage
    size 256 bits double precision floating point YMM registers
    */

    int32_t numrows = A.size();
    int32_t numcols = A[0].size();

    __m128i vec_minus_one= _mm_set1_epi32(-1);
    for(int32_t i=0;i<numrows;i++){
        
        __m256d vec_sum = _mm256_setzero_pd();

        int32_t k=0;
        for(;k<=numcols-4;k+=4){

            __m256d vals = _mm256_loadu_pd(&A[i][k]);
            __m128i inds = _mm_loadu_si128((const __m128i*)&J[i][k]);
            
            __m256d x_vec = _mm256_i32gather_pd(x.data(), inds, 8); //8 is byte scaling, is to skip 8bytes to find next el 
            
            __m128i mask = _mm_cmpeq_epi32(inds, vec_minus_one);
            int bitmask = _mm_movemask_ps(_mm_castsi128_ps(mask));
            if(bitmask!=0) break;

            vec_sum = _mm256_fmadd_pd(vals, x_vec,vec_sum);
        }

        double four_sum_array[4];
        _mm256_storeu_pd(four_sum_array,vec_sum);
        double final_sum_array = four_sum_array[0] + four_sum_array[1] + four_sum_array[2] + four_sum_array[3];

        for(;k<=numcols;k++) final_sum_array += A[i][k]*x[J[i][k]];

        y[i] = final_sum_array;
    }

    return y;
}