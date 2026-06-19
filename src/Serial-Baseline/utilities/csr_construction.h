#pragma once
#include <iostream>
#include <vector>
#include "./../utilities/graph_io.h"
#include "./../utilities/graph_type.h"
#include "./../utilities/csr_type.h"

// #include "./../utilities/csr_construction.h"
using namespace std;

void Contruction_CSR(const Graph& adj_list, csrt& res){
    /*
    #343fff as we are using bidirectional directed graphs 
    value: weights of the edges:
    col_idx: the targt page/node ids for each link
    row_ptr: map pointers indicating where each node's outgoing links start and end in the col_ind array.
    */

    res.num_rows = adj_list.size();
    int total_edges =0;
    res.row_ptr.resize(res.num_rows+1,0);
    
    for(int i=0;i<res.num_rows;i++){
        res.row_ptr[i] = total_edges;
        total_edges += adj_list[i].size();
    }

    res.row_ptr[res.num_rows] = total_edges;
    res.col_idx.resize(total_edges);
    res.value.resize(total_edges);   
    
    // tracking index
    int edge_idx =0;
    for(int i=0;i<res.num_rows;i++){
        vector<int> d(adj_list[i].size());
        for(int el:adj_list[i]){
            res.col_idx[edge_idx] = el;
            if(d[i] > 0 ) res.value[edge_idx] = 1.0/d[i];
            else res.value[edge_idx] = 0.0;
            edge_idx++;
        }
    }

    return;
}
