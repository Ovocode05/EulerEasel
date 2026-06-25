#pragma once
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<tuple>
#include<string>
using namespace std;

tuple<int32_t, int32_t, int32_t> matrix_dim(){
    ifstream file("/home/fakeheadset/Projects/EulerEasel/Data/synthetic/uniform.mtx");
    if(!file.is_open()){
        cerr<<"file error"<<endl;
        return {0,0,0};
    }

    string line="";
    int32_t numrows =0;
    int32_t numcols=0;
    int32_t nnz=0;

    while(getline(file, line)){
        if(line.empty() || line[0]=='%') continue;

        stringstream ss(line);
        if(ss >> numrows >> numcols >> nnz) break;
    }

    file.close();

    return {numrows, numcols, nnz};
}