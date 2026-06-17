#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include "./../utilities/graph_io.h"
#include "./../utilities/graph_type.h"
#include "./../utilities/file_parser.h"
using namespace std;

/*
Pagerrank outputs the probability distribution used to represent the likelihood that a person randomly clicking on 
links will arrive at any particular page.
It is the eiganvalue problem
*/

void pagerRanker(const vector<vector<int>>& adj_list, const int& iters=100, const double& d=0.85){
    int n = adj_list.size();
    if (n == 0) return;

    vector<double> rank(n, 1.0 / n); 
    
    for(int i = 0; i < iters; i++){
        vector<double> new_rank(n, (1.0 - d) / n);

        for(int j = 0; j < n; j++){
            int out = adj_list[j].size();
            
            if(out == 0) {
                for(int k = 0; k < n; k++) {
                    new_rank[k] += d * (rank[j] / n);
                }
                continue;
            }

            double contribution = d * (rank[j] / out);
            
            for(int el : adj_list[j]){
                if (el >= 0 && el < n) {
                    new_rank[el] += contribution;
                }
            }
        }
        rank = move(new_rank); 
    }

    return;
}

int main(int argc, char* argv[]){
        //local variable but type taken directly from "graph_type.h"
    Graph adj_list;
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <start_node> <end_node>" << std::endl;
        return -1;
    }

    int iters=stoi(argv[1]);
    double d=stod(argv[2]);
    
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
    pagerRanker(adj_list, 100, 0.85);
    auto end = chrono::steady_clock::now();
    auto duration_PagerRank = chrono::duration_cast<chrono::milliseconds>(end-start);
    cout<<"time taken to run PagerRank = " <<duration_PagerRank.count()<<endl;


    return 0;
}
