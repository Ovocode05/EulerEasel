#pragma once
#include <iostream>
#include <vector>
using namespace std;

struct csrt{
    int num_rows;
    int num_edges;
    vector<double> value;
    vector<int> row_ptr;
    vector<int> col_idx;
};
