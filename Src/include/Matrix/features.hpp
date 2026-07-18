#include <iostream>
#include <vector>
using namespace std;

struct ScaleFeatures
{
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
};

struct RowDistributionFeatures
{
    double mean = 0.0;
    double cv = 0.0;
    int32_t max = 0;

    double gini = 0.0;

    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
};

struct ELLFeatures
{
    double efficiency = 0.0;
};

struct LocalityFeatures
{
    double avg_column_gap = 0.0;
};

struct HybridFeatures
{
    double tail_row_fraction = 0.0;
    double tail_work_fraction = 0.0;
};

struct MatrixFeatures
{
    ScaleFeatures scale;
    RowDistributionFeatures row_distribution;
    ELLFeatures ell;
    LocalityFeatures locality;
    HybridFeatures hybrid;
};
