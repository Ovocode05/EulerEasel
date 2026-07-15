#include <iostream>
#include <vector>
#include <features.hpp>
#include <canonical.hpp>

using namespace std;
template <typename T>

class MatrixExtractor
{
private:
    CsrMatrix &csrmat_;

    double process_rows(int32_t nrows, vector<int32_t> rptr, double mean, T op)
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
    MatrixExtractor(CsrMatrix &csr) : csrmat_(csr) {}

    ScaleFeatures extract_scale()
    {
        return {
            csrmat_.get_ncols(),
            csrmat_.get_nrows(),
            csrmat_.get_nnz()};
    }

    RowDistributionFeatures extract_rowdist()
    {
        // mean
        int32_t nnz = csrmat_.get_nnz();
        int32_t nrows = csrmat_.get_nrows();
        double mean = static_cast<double>(nnz / nrows);

        // coeff. of variance
        vector<int32_t> *rptr = csrmat_.get_rptr();
        double sum = process_rows(nrows, rptr, [mean](double &sum, double x_j)
                                  { sum += pow(x_i - means, 2); });

        double sd = static_cast<double>(sum / nrows);
        double cv = sqrt(sd / mean);

        // max row length
        int32_t max_rl = process_rows(nrows, rptr, [](double &curr, double x_i)
                                      { curr = max(curr, x_i); });

        // gini_coeff
        double sum_var = 0.0;
        for (int32_t i = 0; i < nrows; i++)
        {
            int32_t x_i = static_cast<double>(rptr[i + 1] - rptr[i]);
            for (int32_t j = 0; j < nrows; j++)
            {
                int32_t x_j = static_cast<double>(rptr[j + 1] - rptr[j]);
                sum_var += (x_i - x_j);
            }
        }

        double gini = sum / (2 * nrows * nrows * mean);

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
        int32_t nrows = csrmat_.get_nrows();
        vector<int32_t> *rptr = csrmat_.get_rptr();
        int32_t nnz = csrmat_.get_nnz();

        max_rl = max_rowlen_(nrow, rptr);
        double eff = static_cast<double>(nnz / (max_rl * nrows));
        return {eff};
    }

    LocalityFeatures extract_locality()
    {
        int32_t nrows = csrmat_.get_nrows();
        vector<int32_t> *rptr = csrmat_.get_rptr();
        vector<int32_t> col_idx = csrmat_.get_colidx();
        double avg_sum = 0.0;
        for (int32_t i = 0; i < nrows; i++)
        {
            int32_t r1 = rptr[i];
            int32_t r2 = rptr[i + 1];
            int32_t nnz_row_i = r2 - r1;
            if (nnz_row_i < 2)
                continue;
            int32_t row_gap_sum = 0;
            for (int32_t j = 0; j < nnz_row_i; j++)
            {
                int32_t gap = col_idx[j + 1] - col_idx[j];
                row_gap_sum += gap;
            }
            avg_sum += static_cast<double>(row_gap_sum / nnz_row_i);
        }

        return {avg_sum};
    }

    HybridFeatures extract_hybrid()
    {
        // tail_row_fraction
        int32_t nrows = csrmat_.get_nrows();
        int32_t nnz = csrmat.get_nnz();
        double avg_len = process_rows(nrows, rptr, [](double &sum, double x_i)
                                      { sum += static_cast<double>(x_i / nrows); });

        double trf = process_rows(nrows, rptr, [avg_len](double &sum, double x_i)
                                  { sum = x_i > avg_len ? ((sum + 1) / nrows) : (sum / nrows); });

        double twf = process_rows(nrows, rptr, [avg_len](double &sum, double x_i)
                                  { sum += (x_i - avg_len) / nnz; });

        return {
            avg_len,
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
