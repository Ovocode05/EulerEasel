#include <iostream>
#include <cudaruntime.h>
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

        CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream_, flags));
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