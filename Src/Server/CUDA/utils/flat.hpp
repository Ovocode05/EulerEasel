#include <iostream>
#include <vector>
using namespace std;

pair<vector<double>, vector<int32_t>> flatten(
    const vector<vector<int32_t>> J,
    const vector<vector<double>> A)
{
    int32_t r = A.size();
    int32_t c = A[0].size();
    vector<double> A_flat(r * c);
    vector<int32_t> J_flat(r * c);

    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            A_flat[j * r + i] = A[i][j];
            J_flat[j * r + i] = J[i][j];
        }
    }

    return {A_flat, J_flat};
}