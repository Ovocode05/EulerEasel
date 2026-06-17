#include <iostream>
#include <fstream> 
#include <string>
#include <vector>
#include <charconv>
#include <string_view>
#include <ranges>
#include <sstream>
#include <queue>
#include "./../utilities/graph_io.h"
using namespace std;

//#343fff getting the file access and parsing once
int file_access(Graph& adj_list){
    string cache_path = "/home/fakeheadset/Projects/EulerEasel/src/SerialBaseline";
    ifstream check_cache(cache_path, ios::binary);
    if(check_cache.good()){
        check_cache.close();
        cout<<"Cache found"<<endl;
        load_graph_structure(adj_list, cache_path);
        return 0;
    }
    check_cache.close();
    cout<<"No cache file found"<<endl;
    ifstream inputFile("/home/fakeheadset/Projects/EulerEasel/Data/graph-test.txt");
    
    if(!inputFile.is_open()){
        cerr<<"this file couldn't open"<<endl;
        return -1;
    }

    string dummyline;
    for(int i=0;i<5;i++){
        if(!getline(inputFile, dummyline)) return -1;
    }

    string line;
    vector<int> numbers;

    while(getline(inputFile, line)){
        // Ignore empty lines or comments
        if(line.empty() || line[0] == '#') continue;

        // Use stringstream to naturally extract tab/space separated tokens
        stringstream ss(line);
        int u = 0, v = 0;
        
        // Extract both integers directly. Whitespace/tabs are handled natively.
        if (ss >> u >> v) {
            // Dynamically resize your adj_list vector if a node ID exceeds its capacity
            int max_node = max(u, v);
            if (max_node >= adj_list.size()) {
                adj_list.resize(max_node + 1);
            }

            // Populate the undirected graph safely
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
    }   

    inputFile.close();

    cout<<"saving graph structure to binary cache for future runs"<<endl;
    save_graph_struture(adj_list, cache_path);

    return 0;
}