#include <iostream>
#include <vector>
#include <chrono>
#include "./../utilities/graph_io.h"
#include "./../utilities/graph_type.h"
#include "./../utilities/csr_construction.h"
#include "./../utilities/spMV_kernel.h"
#include "./../utilities/file_parser.h"
#include "./../utilities/csr_type.h"
using namespace std;


void power_iteration(const csrt& res){
    double d = 0.85;
    vector<int> x_curr(res.num_rows, 1.0/res.num_rows);
    vector<int> x_next(res.num_rows);
    double C = (1.0 - d)/res.num_rows;
    int loss=0;
    double eps;

    x_next = SpMV_kernel(res.row_ptr, res.col_idx, res.value, x_curr, x_next, res.num_rows);

    for(int i=0;i<res.num_rows;i++){
        if(loss>eps){
            x_next[i] = (d * x_next[i] + C);
            loss += abs(x_next[i] - x_curr[i]);
            cout<<"loss at iteration "<<i<<"="<<loss<<endl;
        }else{
            cout<<"Convereged"<<endl;
            break;
        }
    }

    swap(x_curr, x_next);
}

int main(int argc, char* argv[]){
    Graph adj_list;
    csrt res;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <start_node> <end_node>" << std::endl;
        return -1;
    }

    int iters=stoi(argv[1]);
    double d=stod(argv[2]);

    auto start = chrono::steady_clock::now();
    if(file_access(adj_list)!=0){
        cerr <<"error in opening file"<<endl;
        return -1;
    }

    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start); 
    cout << "time taken to convert and load the file" << duration.count() <<endl;

    auto start_computation = chrono::steady_clock::now();
    Contruction_CSR(adj_list, res);
    power_iteration(res);
    auto end_computation = chrono::steady_clock::now();

    auto duration_comp = chrono::duration_cast<chrono::milliseconds>(end_computation -start_computation);
    cout<<"time taken to converge = "<< duration_comp.count()<<endl;


    return 0;
}