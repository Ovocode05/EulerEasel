#pragma once
#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<iomanip>
#include<filesystem>
using namespace std;
namespace fs = std::filesystem;

void create_outfile(string dir, string filename, const vector<double>& results){
    fs::path outputDir = dir;
    fs::path filePath = outputDir / filename; 

    ofstream outfile(filePath);
    if(!outfile.is_open()){
        cerr<<"this file is not working"<<endl;
        return;
    }

    outfile << fixed << setprecision(10);
    for(const double& el: results){
        outfile << el<<"\n";
    }
    
    outfile.close();
    cout<<"file has been written and saved"<<endl;
    return;
}