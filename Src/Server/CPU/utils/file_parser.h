#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<algorithm>
#include "./../utils/datatype.h"
using namespace std;

int file_parser(vector<matrix_el>& matrix){

    //get the file
    ifstream file("/home/fakeheadset/Projects/EulerEasel/Data/synthetic/skewed.mtx");
    if(!file.is_open()){
        cerr<<"couldn't open the file"<<endl;
        return -1;
    }

    string line;
    bool first_data_line = false;

    //get the rest line and parse the data
    while(getline(file, line)){
        if(line=="" || line[0]=='%'){
            continue;
        }

        if(!first_data_line){
            first_data_line=true;
            continue; 
        } 
        
        //load the string into the stream
        stringstream ss(line);
        int rows;
        int cols;
        double vals;
        if(ss >> rows >> cols >> vals){
            matrix_el el;
            el.row_el = rows-1;
            el.col_el = cols-1;
            el.val_el = vals;
            matrix.push_back(el);
        }
    }

    //lambda function
    sort(matrix.begin(), matrix.end(), [](const matrix_el& a, const matrix_el& b){
        if(a.row_el!=b.row_el) return a.row_el < b.row_el;
        return a.col_el < b.col_el;
    });

    return 0;
}