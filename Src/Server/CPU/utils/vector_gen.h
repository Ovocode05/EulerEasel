#include<iostream>
#include<random>
#include<vector>
#include "./matrix_dim.h"

class Central_Vector{
public:
    static vector<double> generate(){
    auto [r,c,nnz] = matrix_dim();
    std::random_device rd;

    //identical seed number generates same values for instances
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1, 1000);

    vector<double> x;
    x.reserve(c);
    for (size_t i = 0; i < c; ++i) {
        x.emplace_back(distr(gen));
    }
    return x;
}
};