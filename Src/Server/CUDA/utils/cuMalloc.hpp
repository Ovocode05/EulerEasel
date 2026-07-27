#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "./err.hpp"
using namespace std;
template <typename T>

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
        cudaMalloc(&ptr_, size_ * sizeof(T));
        CUDA_CHECK();
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
        cudaMemcpyAsync(ptr_, host, sizeof(T) * count, cudaMemcpyHostToDevice, stream);
        CUDA_CHECK();
    }

    void device_to_host(
        T *host,
        size_t count,
        cudaStream_t stream = 0) const
    {
        cudaMemcpyAsync(host, ptr_, sizeof(T) * count, cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK();
    }

    deviceBuffer(const deviceBuffer &) = delete;
    deviceBuffer &operator=(const deviceBuffer &) = delete;
};