#include <iostream>
#include <cuda_runtime.h>
#include "./err.hpp"
using namespace std;

class CudaStream
{
private:
    cudaStream_t stream_;

public:
    explicit CudaStream(bool non_blocking = true)
    {
        unsigned int flags =
            non_blocking
                ? cudaStreamNonBlocking
                : cudaStreamDefault;

        cudaStreamCreateWithFlags(&stream_, flags);
        CUDA_CHECK();
    }

    ~CudaStream()
    {
        cudaStreamDestroy(stream_);
    }

    cudaStream_t get() const
    {
        return stream_;
    }

    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;
};