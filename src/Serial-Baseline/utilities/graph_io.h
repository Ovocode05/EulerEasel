#pragma once
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <string>
#include <vector>
#include "./../utilities/graph_type.h"
using namespace std;

inline void save_graph_struture(const Graph& adj_list, const string& cache_path){
    ofstream out(cache_path,  ios::binary);
    if(!out) return;

    size_t map_size = adj_list.size();
    out.write(reinterpret_cast<const char*> (&map_size), sizeof(map_size));

    for(const auto& neighbors: adj_list){
        size_t neighbor_count = neighbors.size();
        out.write(reinterpret_cast<const char*> (&neighbor_count), sizeof(neighbor_count));

        if(neighbor_count>0) out.write(reinterpret_cast<const char*>(neighbors.data()), neighbor_count*sizeof(int));
    }
}

inline void load_graph_structure(Graph& adj_list, const string& cache_path){
    ifstream in(cache_path, ios::binary);
    if(!in) return;

    size_t map_size;
    in.read(reinterpret_cast<char*>(&map_size),sizeof(map_size));
    adj_list.resize(map_size);

    for(size_t i=0;i<map_size;i++){
        size_t neighbor_count;
        in.read(reinterpret_cast<char*>(&neighbor_count), sizeof(neighbor_count));

        auto& neighbors = adj_list[i];
        neighbors.resize(neighbor_count);

        if(neighbor_count>0) in.read(reinterpret_cast<char*>(neighbors.data()), neighbor_count*sizeof(int));
    }
}