#pragma once
#include<iostream>
#include<random>
#include<vector>
#include "./matrix_dim.h"

class Central_Vector{
public:
    static vector<double> generate(int32_t r, int32_t c, int32_t nnz){
    std::random_device rd;

    //identical seed number generates same values for instances
    std::mt19937 gen(42);
    std::uniform_int_distribution<> distr(1, 1000);

    vector<double> x;
    x.reserve(c);
    for (int32_t i = 0; i < c; ++i) {
        x.emplace_back(distr(gen));
    }
    return x;
}
};