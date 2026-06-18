#include <iostream>
#include <vector>
#include "./../utilities/graph_io.h"
#include "./../utilities/graph_type.h"
using namespace std;

struct csr{
    vector<int> value;
    vector<int> row_ptr;
    vector<int> col_idx;
};

csr CSR_contruct(const vector<vector<int>>& adj_list){
    /*
    #343fff as we are using bidirectional directed graphs 
    value: weights of the edges:
    col_idx: the targt page/node ids for each link
    row_ptr: map pointers indicating where each node's outgoing links start and end in the col_ind array.
    */
}
