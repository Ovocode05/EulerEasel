#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

struct edge{
    int from;
    int to;
};

int main(){
    string filename = "graph-test.txt";
    ofstream output(filename);

    if(!output.is_open()){
        cerr<<"Error opening file"<<endl;
        return 1;
    }
    
    output<< "#Undirected graph: ../../data/output/lj.ungraph.txt\n";
    output<<"# LiveJournal\n";
    output<<"# Nodes: 10 Edges: 11\n";
    output<<"# FromNodeId	ToNodeId\n";

    vector<edge> design = {{0,1},{1,2},{1,3},{2,4},{2,5},{3,6},{3,7},{5,8},{6,8},{6,9},{8,10},{9,10}};
    for(auto& el:design) output<<el.from<<'\t'<<el.to<<'\n';

    output.close();
    cout<<"file has been written"<<endl;
    return 0;
    
}