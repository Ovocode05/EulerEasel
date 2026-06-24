#pragma once
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<tuple>
#include<string>
using namespace std;

tuple<int, int, int> matrix_dim(){
    ifstream file("/home/fakeheadset/Projects/EulerEasel/Data/synthetic/uniform.mtx");
    if(!file.is_open()){
        cerr<<"file error"<<endl;
        return {0,0,0};
    }

    string line="";
    int numrows =0;
    int numcols=0;
    int nnz=0;

    while(getline(file, line)){
        if(line.empty() || line[0]=='%') continue;

        stringstream ss(line);
        if(ss >> numrows >> numcols >> nnz) break;
    }

    file.close();

    return {numrows, numcols, nnz};
}