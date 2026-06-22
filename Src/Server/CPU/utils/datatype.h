#include <iostream>
using namespace std;

struct matrix_el{
    int row_el;
    int col_el;
    double val_el;
};

struct CSR {
    vector<int> rptr;
    vector<int> ind;
    vector<double> vals;
    int num_rows; 
};

struct ell{

};