#include <iostream>
#include <vector>
#include <cmath>
#include "./features.hpp"
#include "./canonical.hpp"

using namespace std;
template <typename T>

class MatrixExtractor
{
private:
    CsrMatrix<T> &csrmat_;
    template <typename Op>
    double process_rows(int32_t nrows, const vector<int32_t> &rptr, Op op)
    {
        double sum = 0.0;
        for (int i = 0; i < nrows; i++)
        {
            int32_t r1 = rptr[i];
            int32_t r2 = rptr[i + 1];
            double x_i = static_cast<double>(r2 - r1);

            op(sum, x_i);
        }

        return sum;
    }

public:
    MatrixExtractor(CsrMatrix<T> csrmat) : csrmat_(csrmat) {}

    ScaleFeatures extract_scale()
    {
        return {
            csrmat_.get_ncol(),
            csrmat_.get_nrow(),
            csrmat_.get_nnz()};
    }

    RowDistributionFeatures extract_rowdist()
    {
        // mean
        int32_t nnz = csrmat_.get_nnz();
        int32_t nrows = csrmat_.get_nrow();
        double mean = static_cast<double>(nnz / nrows);

        // coeff. of variance
        const vector<int32_t> &rptr = csrmat_.get_rptr();
        double summation = process_rows(nrows, rptr, [mean](double &sum, double x_i)
                                        { sum += pow(x_i - mean, 2); });

        double sd = static_cast<double>(summation / nrows);
        double cv = sqrt(sd / mean);

        // max row length
        int32_t max_rl = process_rows(nrows, rptr, [](double &curr, double x_i)
                                      { curr = max(curr, x_i); });

        // gini_coeff
        double sum_var = 0.0;
        for (int32_t i = 0; i < nrows; i++)
        {
            int32_t r1 = rptr[i];
            int32_t r2 = rptr[i + 1];
            double x_i = static_cast<double>(r2 - r1);
            for (int32_t j = 0; j < nrows; j++)
            {
                int32_t r3 = rptr[i];
                int32_t r4 = rptr[i + 1];
                double x_j = static_cast<double>(r4 - r3);
                sum_var += (x_i - x_j);
            }
        }

        double gini = summation / (2 * nrows * nrows * mean);

        // p50, p90, p99
        double p50 = 0.50 * nrows;
        double p90 = 0.90 * nrows;
        double p99 = 0.99 * nrows;

        return {
            mean,
            cv,
            max_rl,
            gini,
            p50,
            p90,
            p99};
    }

    ELLFeatures extract_ell()
    {
        int32_t nrows = csrmat_.get_nrow();
        const vector<int32_t> &rptr = csrmat_.get_rptr();
        int32_t nnz = csrmat_.get_nnz();

        int32_t max_rl = static_cast<int32_t>(process_rows(nrows, rptr, [](double &curr, double x_i)
                                                           { curr = max(curr, x_i); }));
        double eff = static_cast<double>(nnz / (max_rl * nrows));

        return {eff};
    }

    LocalityFeatures extract_locality()
    {
        int32_t nrows = csrmat_.get_nrow();
        const vector<int32_t> &rptr = csrmat_.get_rptr();
        const vector<int32_t> &col_idx = csrmat_.get_colidx();
        double avg_sum = 0.0;
        for (int32_t i = 0; i < nrows; i++)
        {
            if (i + 1 >= rptr.size())
            {
                break;
            }

            int32_t r1 = rptr[i];
            int32_t r2 = rptr[i + 1];
            int32_t nnz_row_i = r2 - r1;

            if (nnz_row_i < 2)
                continue;

            int32_t row_gap_sum = 0;
            for (int32_t j = r1; j < r2 - 1; j++)
            {
                if (j + 1 >= col_idx.size())
                {
                    break;
                }

                int32_t gap = col_idx[j + 1] - col_idx[j];
                row_gap_sum += gap;
            }
            int32_t total_gaps = nnz_row_i - 1;

            avg_sum += static_cast<double>(row_gap_sum) / total_gaps;
        }

        return {avg_sum};
    }

    HybridFeatures extract_hybrid()
    {
        // tail_row_fraction
        int32_t nrows = csrmat_.get_nrow();
        int32_t nnz = csrmat_.get_nnz();
        const vector<int32_t> &rptr = csrmat_.get_rptr();
        double avg_len = process_rows(nrows, rptr, [nrows](double &sum, double x_i)
                                      { sum += static_cast<double>(x_i / nrows); });

        double trf = process_rows(nrows, rptr, [avg_len, nrows](double &sum, double x_i)
                                  { sum = x_i > avg_len ? ((sum + 1) / nrows) : (sum / nrows); });

        double twf = process_rows(nrows, rptr, [avg_len, nnz](double &sum, double x_i)
                                  { sum += (x_i - avg_len) / nnz; });

        return {
            trf,
            twf};
    }

    MatrixFeatures extract_all()
    {
        ScaleFeatures s = extract_scale();
        RowDistributionFeatures r = extract_rowdist();
        ELLFeatures e = extract_ell();
        LocalityFeatures l = extract_locality();
        HybridFeatures h = extract_hybrid();

        return {
            s, r, e, l, h};
    }
};

// int main()
// {
//     // set up
//     CSR raw_csr;
//     raw_csr.rptr = {0, 2, 3, 5};              // Row pointers (4 elements for 3 rows)
//     raw_csr.ind = {0, 2, 1, 0, 2};            // Column indices for non-zero elements
//     raw_csr.vals = {1.0, 2.0, 3.0, 4.0, 5.0}; // The actual values
//     raw_csr.num_rows = 3;

//     int32_t total_columns = 3;

//     // 2. Wrap the raw data inside the CsrMatrix class
//     // We instantiate it with <double> to match the raw_csr.vals type
//     CsrMatrix<double> matrix(raw_csr, total_columns);

//     // 3. Pass the matrix wrapper into the MatrixExtractor
//     MatrixExtractor<double> extractor(matrix);

//     // 4. Run the calculation
//     // This will process the 3 rows using the internal logic and operation
//     MatrixFeatures res = extractor.extract_all();

//     // 5. Output the calculated result
//     cout << "The calculated extraction sum is done" << endl;

//     return 0;
// }