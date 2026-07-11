#include <iostream>
#include <cudaruntime.h>
#include <vector>
template <typename T>
using namespace std;

/*
explains what is in this file
*/

class deviceBuffer
{
private:
    T *ptr_;
    size_t size_;

public:
    deviceBuffer() = default;

    explicit deviceBuffer(size_t size) : size_(size)
    {
        CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(T)));
    }

    ~deviceBuffer()
    {
        if (ptr_)
            cudaFree(ptr_);
    }

    T *data()
    {
        return ptr_;
    }

    const T *data() const
    {
        return ptr_;
    }

    size_t size() const
    {
        return size_;
    }

    void host_to_device(
        const T *host,
        size_t count,
        cudaStream_t stream = 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(host, ptr_, sizeof(T) * count, cudaMemcpyHostToDevice, stream));
    }

    void device_to_host(
        T *host,
        size_t count,
        cudaStream_t stream = 0) const
    {
        CUDA_CHECK(cudaMemcpyAsync(host, ptr_, sizeof(T) * count, cudaMemcpyDeviceToHost, stream));
    }

    deviceBuffer(const deviceBuffer &) = delete;
    deviceBuffer &operator=(const deviceBuffer &) = delete;
};