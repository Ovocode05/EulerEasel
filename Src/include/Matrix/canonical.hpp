#include <iostream>
#include <vector>
#include "./../../Server/utils/datatype.h"
using namespace std;
template <typename T>

class CsrMatrix
{
private:
    CSR csr_;
    int32_t num_col_;

public:
    CsrMatrix(CSR &csrmat, int32_t col) : csr_(csrmat), num_col_(col)
    {
        if (csr_.rptr.size() != static_cast<size_t>(csr_.num_rows + 1))
        {
            throw runtime_error("CSR Initialization Error: rptr size must be num_rows + 1.");
        }

        if (csr_.rptr.back() != static_cast<int32_t>(csr_.ind.size()))
        {
            throw runtime_error("CSR Initialization Error: The last element of rptr must match the total number of non-zeros (ind.size()).");
        }

        if (csr_.ind.size() != csr_.vals.size())
        {
            throw runtime_error("CSR Initialization Error: Mismatch between non-zero indices and values array sizes.");
        }
    }

    // functions to access private values
    const vector<int32_t> &get_rptr() const
    {
        return csr_.rptr;
    }

    const vector<int32_t> &get_colidx() const
    {
        return csr_.ind;
    }

    const vector<T> &get_vals() const
    {
        return csr_.vals;
    }

    int32_t get_ncol()
    {
        return num_col_;
    }

    int32_t get_nrow()
    {
        return csr_.num_rows;
    }

    int32_t get_nnz()
    {
        return csr_.vals.size();
    }
};
