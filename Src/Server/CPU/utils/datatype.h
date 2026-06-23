#pragma once
#include <iostream>
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
    vector<unsigned int> col_ind;
    vector<double> val;
};