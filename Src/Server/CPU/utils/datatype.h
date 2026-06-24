#pragma once
#include <iostream>
#include<vector>
#include "./matrix_dim.h"
using namespace std;

struct matrix_el{
    unsigned int row_el;
    unsigned int col_el;
    double val_el;
};

struct CSR {
    vector<unsigned int> rptr;
    vector<unsigned int> ind;
    vector<double> vals;
    unsigned int num_rows; 
};

struct ell{
    unsigned int numcols;
    unsigned int numrows;
    unsigned int max_padd;
    vector<vector<unsigned int>> col_ind;
    vector<vector<double>> val;
};

struct hybd{
    ell el_part;
    CSR csr_part;
    vector<matrix_el> ell_entries;
    vector<matrix_el> csr_entries;
};