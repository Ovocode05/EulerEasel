    #pragma once
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

    vector<double> SpMv_kernel_ell(const vector<double>& x, const vector<vector<double>>& A, const vector<vector<int32_t>>& J){
        /*
        SpMV kernel of Ellpack has shown more efficient results for GPU and parallel computations
        */
        
        int32_t numrows = A.size();
        int32_t numcols = A[0].size();
        vector<double> y(numrows,0);

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

    vector<double> ell_spMV_AVX(const vector<double>& x, const vector<vector<double>>& A, const vector<vector<int32_t>>& J){
        /*
        The AVX intrinsic implemented for the SpMV kernel for csr format 
        int32_t bits variable storage
        size 256 bits double precision floating point YMM registers
        */

        int32_t numrows = A.size();
        int32_t numcols = A[0].size();
        vector<double> y(numrows, 0);

        __m128i vec_minus_one= _mm_set1_epi32(-1);

        #pragma parallel for default(none) shared(numrows, numcols, y, x, A, J) schedule(runtime)
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

    vector<double>ell_pack_AVX_vertical(const vector<double>& x, const vector<vector<double>>& A, const vector<vector<int32_t>>& J){
        
        int32_t numrows = A.size();
        int32_t maxpadd = A[0].size();
        vector<double> y(numrows, 0);
        vector<int32_t> inds;
        vector<double> vals; 

        vals.reserve(numrows * maxpadd);
        inds.reserve(numrows * maxpadd);

        for(int32_t j=0;j<numrows;j++){
            for(int32_t l=0;l<maxpadd;l++){
                //rowmajor
                vals.emplace_back(A[j][l]);
                inds.emplace_back(J[j][l]);
            }
        }

        #pragma parallel for default(none) shared(numrows, maxpadd, y, x, A, J) schedule(runtime)
        //safe auto privatization for openMP
        for(int32_t i=0 ;i<=numrows-4;i+=4){

            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1  = _mm256_setzero_pd();
            __m256d sum2  = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();

            //rowmajor  
            int32_t idx0 = i*maxpadd;
            int32_t idx1 = (i+1)*maxpadd;  
            int32_t idx2 = (i+2)*maxpadd;
            int32_t idx3 = (i+3)*maxpadd; 
            
            int32_t k=0;
            for(;k<=maxpadd-4; k+=4){

                //force cpu to break it into smaller instruction - wastage of CPU cyles.
                // __m256d val = _mm256_set_pd(vals[idx3], vals[idx2] ,vals[idx1], vals[idx0]);
                // __m128i ind = _mm_set_epi32(inds[idx3], inds[idx2], inds[idx1], inds[idx0]);

                __m256d val0 = _mm256_loadu_pd(&vals[idx0+k]);
                __m128i ind0 = _mm_loadu_si128((const __m128i*)&inds[idx0+k]);
                __m256d vec_x0 = _mm256_i32gather_pd(x.data(), ind0, 8);
                sum0 = _mm256_fmadd_pd(val0, vec_x0 ,sum0);

                __m256d val1 = _mm256_loadu_pd(&vals[idx1+k]);
                __m128i ind1 = _mm_loadu_si128((const __m128i*)&inds[idx1+k]);
                __m256d vec_x1 = _mm256_i32gather_pd(x.data(), ind1, 8);
                sum1 = _mm256_fmadd_pd(val1, vec_x1 ,sum1);

                __m256d val2 = _mm256_loadu_pd(&vals[idx2+k]);
                __m128i ind2 = _mm_loadu_si128((const __m128i*)&inds[idx2+k]);
                __m256d vec_x2 = _mm256_i32gather_pd(x.data(), ind2, 8);
                sum2 = _mm256_fmadd_pd(val2, vec_x2 ,sum2);

                __m256d val3 = _mm256_loadu_pd(&vals[idx3+k]);
                __m128i ind3 = _mm_loadu_si128((const __m128i*)&inds[idx3+k]);
                __m256d vec_x3 = _mm256_i32gather_pd(x.data(), ind3, 8);
                sum3 = _mm256_fmadd_pd(val3, vec_x3 ,sum3);  
            }

            alignas(32) double temp0[4], temp1[4], temp2[4], temp3[4];
            _mm256_store_pd(temp0, sum0);
            _mm256_store_pd(temp1, sum1);
            _mm256_store_pd(temp2, sum2);
            _mm256_store_pd(temp3, sum3);

            y[i] = temp0[0] + temp0[1] + temp0[2] + temp0[3];
            y[i+1] = temp1[0] + temp1[1] + temp1[2] + temp1[3];
            y[i+2] = temp2[0] + temp2[1] + temp2[2] + temp2[3];
            y[i+3] = temp3[0] + temp3[1] + temp3[2] + temp3[3];

            for (; k < maxpadd; k++) {
                y[i]     += vals[idx0 + k] * x[inds[idx0 + k]];
                y[i + 1] += vals[idx1  + k] * x[inds[idx1 + k]];
                y[i + 2] += vals[idx2 + k] * x[inds[idx2 + k]];
                y[i + 3] += vals[idx3 + k] * x[inds[idx3 + k]];
            }
        }
        
        int32_t remainder = (numrows/4)*4;
        for (int32_t i =remainder; i < numrows; i++) {
            double sum = 0.0;
            int32_t base = i * maxpadd;
            for (int32_t k = 0; k < maxpadd; k++) {
                int32_t idx = base + k;
                int32_t col = inds[idx];
                sum += vals[idx] * x[col]; // Fixed accumulation logic
            }

            y[i] = sum;
        }
        
        return y;
    }
