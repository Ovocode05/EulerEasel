#include <iostream>
#include <vector>
#include "./../utilities/graph_io.h"
#include "./../utilities/graph_type.h"
#include "./../utilities/csr_construction.h"
using namespace std;

vector<int> SpMV_kernel(vector<int> row_ptr, vector<int> col_idx, vector<double> value, vector<int> x, vector<int> y, int n ){

    for(int i=0;i<n;i++){
        int sum=0.0;
        int start = row_ptr[i];
        int end = row_ptr[i+1];
        for(int j=start;j<end;j++){
            int col = col_idx[j];
            sum += value[j]* x[col];
        }
        y[i]=sum;
    }

    return y;
}
