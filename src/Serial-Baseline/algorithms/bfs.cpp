#include <iostream>
#include <fstream> 
#include <string>
#include <vector>
#include <unordered_map>
#include <charconv>
#include <string_view>
#include <ranges>
#include <queue>
#include <chrono>
#include "./../utilities/file_parser.h"
#include "./../utilities/graph_io.h"
#include "./../utilities/graph_type.h"
using namespace std;

/*
dataset: Undirected LiveJournal network (346M+ rows)
source : SNAP (stanford large data collection network)
aim : serial implementation breadth first search (bfs) algorithm 
What questions bfs will be handling: shortest path between 2 users, calulate degee of separation,find all the connected components 
*/

//bfs core function
vector<int> run_bfs(const vector<vector<int>>& adj_list, int start_node, int end_node){
    vector<int> bfs;

    if (start_node < 0 || start_node >= adj_list.size()) {
        cout << "couldn't find start node" << endl;
        return bfs;
    }

    if (start_node == end_node) {
        bfs.push_back(start_node);
        cout << "destination found" << endl;
        return bfs;
    }

    //the sequence in which the graph is travesed
    queue<int> seq;
    seq.push(start_node);

    //visited nodes
    vector<int> vis(adj_list.size(), 0);
    vis[start_node] = 1;

    while(!seq.empty()){
        int node = seq.front();
        seq.pop();
        bfs.push_back(node);

        for(auto it: adj_list[node]){
            if(!vis[it]){
                if (it==end_node){
                    cout<<"destination found"<<endl;
                    return bfs;
                }
                vis[it]=1;
                seq.push(it);
            }
        }
    }
    
    cout<<"process has ended but didn't found any"<<endl;
    return bfs;
}

int main(int argc, char* argv[]){
    //local variable but type taken directly from "graph_type.h"
    Graph adj_list;
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <start_node> <end_node>" << std::endl;
        return -1;
    }

    int start_node=stoi(argv[1]);
    int end_node=stoi(argv[2]);
    
    auto start_load = chrono::steady_clock::now();
    if(file_access(adj_list)!=0){
        cerr<<"Failed to construct graph structure"<<endl;
        return -1;
    }
    auto end_load = chrono::steady_clock::now();
    auto duration_loading = chrono::duration_cast<chrono::milliseconds>(end_load-start_load);
    cout<<"time taken to form data structure = "<<duration_loading.count()<<endl; 

    auto start = chrono::steady_clock::now();
    cout<<"size of the adj_list is ="<< adj_list.size()<<endl;

    // for(const auto& el: adj_list){
    //     for(const auto& vec_el: el){
    //         cout<<vec_el<<" ";
    //     }
    //     cout<<endl;
    // }

    vector<int> bfs = run_bfs(adj_list, start_node, end_node);
    for(auto& el: bfs) {
        cout<<el<<"->";
    }
    cout<<"end"<<endl;


    auto end = chrono::steady_clock::now();
    auto duration_bfs = chrono::duration_cast<chrono::milliseconds>(end-start);
    cout<<"time taken to run bfs = " <<duration_bfs.count()<<endl;


    return 0;
}