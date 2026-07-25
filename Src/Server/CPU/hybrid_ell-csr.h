#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "./../utils/datatype.h"
#include "./../utils/datatype.h"
#include "./../utils/create_file.h"
#include "./../utils/file_parser.h"
#include "./../utils/format/csr_format.h"
#include "./../utils/format/ellpack_format.h"
using namespace std;

tuple<vector<vector<double>>, vector<vector<int>>, CSR> hybrid_format(hybd &hybrid, vector<matrix_el> &matrix, int32_t r, int32_t c, int32_t nnz)
{
    (void)nnz;

    vector<int32_t> row_counter(r, 0);
    for (const auto &el : matrix)
    {
        row_counter[el.row_el]++;
    }

    vector<int32_t> sorted = row_counter;
    sort(sorted.begin(), sorted.end());
    int32_t threshold = sorted[0];
    int32_t max_jump = 0;
    for (size_t i = 1; i < sorted.size(); i++)
    {
        int32_t jump = sorted[i] - sorted[i - 1];
        if (jump > max_jump)
        {
            max_jump = jump;
            threshold = sorted[i - 1];
        }
    }

    for (const auto &el : matrix)
    {
        int32_t row = el.row_el;
        if (row_counter[row] <= threshold)
        {
            // apply csr
            hybrid.ell_entries.emplace_back(el);
        }
        else
        {
            // apply ell
            hybrid.csr_entries.emplace_back(el);
        }
    }

    // measure r,c,nnz for each part of the matrix
    int32_t nnz_csr = hybrid.csr_entries.size();
    int32_t nnz_ell = hybrid.ell_entries.size();

    Csrformat(hybrid.csr_entries, r, c, nnz_csr, hybrid.csr_part);
    auto [A, J] = ellpack_format(hybrid.ell_entries, r, c, nnz_ell, hybrid.el_part);

    return {A, J, hybrid.csr_part};
}

inline tuple<vector<double>, double> SpMv_kernel_hybrid(hybd &hybrid, const vector<double> &x, vector<vector<double>> &A, vector<vector<int32_t>> &J, int32_t r)
{
    auto [y_csr, nano1] = SpMv_kernel(hybrid.csr_part, x);
    auto [y_ell, nano2] = SpMv_kernel_ell(x, A, J);
    vector<double> y_new(r, 0);
    const auto start = chrono::steady_clock::now();

    for (int32_t i = 0; i < r; i++)
    {
        y_new[i] = y_csr[i] + y_ell[i];
    }

    const auto end = chrono::steady_clock::now();
    double rntime_ns = chrono::duration<double, nano>(end - start).count();
    return {move(y_new), rntime_ns + nano1 + nano2};
}

inline tuple<vector<double>, double> SpMv_kernel_hybrid_x4(hybd &hybrid, const vector<double> &x, vector<vector<double>> &A, vector<vector<int32_t>> &J, int32_t r)
{
    auto [y_csr, nano1] = SpMV_kernel_AVX(hybrid.csr_part, x);
    auto [y_ell, nano2] = ell_spMV_AVX(x, A, J);
    vector<double> y_new(r, 0);
    const auto start = chrono::steady_clock::now();

    for (int32_t i = 0; i < r; i++)
    {
        y_new[i] = y_csr[i] + y_ell[i];
    }

    const auto end = chrono::steady_clock::now();
    double rntime_ns = chrono::duration<double, nano>(end - start).count();
    return {move(y_new), rntime_ns + nano1 + nano2};
}

inline tuple<vector<double>, double> SpMv_kernel_hybrid_16x(hybd &hybrid, const vector<double> &x, vector<vector<double>> &A, vector<vector<int32_t>> &J, int32_t r)
{
    auto [y_csr, nano1] = SpMV_kernel_AVX(hybrid.csr_part, x);
    auto [y_ell, nano2] = ell_pack_AVX_vertical(x, A, J);
    vector<double> y_new(r, 0);
    const auto start = chrono::steady_clock::now();

    for (int32_t i = 0; i < r; i++)
    {
        y_new[i] = y_csr[i] + y_ell[i];
    }

    const auto end = chrono::steady_clock::now();
    double rntime_ns = chrono::duration<double, nano>(end - start).count();
    return {move(y_new), rntime_ns + nano1 + nano2};
}