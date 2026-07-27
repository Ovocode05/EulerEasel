#include <iostream>
#include <cuda_runtime.h>
#include "./err.hpp"
using namespace std;
template <typename T>

class cudaRegister
{
private:
    void *ptr_;
    size_t size_;

public:
    cudaRegister(
        size_t size,
        uintptr_t ptr) : size_(size), ptr_(reinterpret_cast<void *>(ptr))
    {
        cudaHostRegister(
            ptr_,
            size_,
            cudaHostRegisterDefault);
        CUDA_CHECK();
    }

    ~cudaRegister()
    {
        cudaHostUnregister(ptr_);
    }

    cudaRegister(const cudaRegister &) = delete;
    cudaRegister &operator=(const cudaRegister &) = delete;
};