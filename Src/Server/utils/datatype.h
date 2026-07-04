#pragma once
#include <iostream>
#include<vector>
#include "./matrix_dim.h"
using namespace std;

struct matrix_el{
    int32_t row_el;
    int32_t col_el;
    double val_el;
};

struct CSR {
    vector<int32_t> rptr;
    vector<int32_t> ind;
    vector<double> vals;
    int32_t num_rows; 
};

struct ell{
    int32_t numcols;
    int32_t numrows;
    int32_t max_padd;
    vector<vector<int32_t>> col_ind;
    vector<vector<double>> val;
};

struct hybd{
    ell el_part;
    CSR csr_part;
    vector<matrix_el> ell_entries;
    vector<matrix_el> csr_entries;
};