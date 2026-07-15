#include <iostream>
#include <vector>
using namespace std;
template <typename T>

class CsrMatrix
{
private:
    vector<int32_t> rtrp_;
    vector<int32_t> colsidx_;
    vector<T> vals_;
    int32_t rows_;
    int32_t cols_;
    int32_t nnz;

public:
    // functions to access private values
    const vector<int32_t> &get_rptr() const
    {
        return rptr_;
    }

    const vector<int32_t> &get_colidx() const
    {
        return colsidx_;
    }

    const vector<T> &get_vals() const
    {
        return vals;
    }

    int32_t get_ncol()
    {
        return cols_;
    }

    int32_t get_nrow()
    {
        return rows_;
    }

    int32_t get_nnz()
    {
        return nnz_;
    }
};
