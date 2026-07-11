#include <iostream>
#include <cudaruntime.h>
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
        void *ptr) : size_(size), ptr_(ptr)
    {
        CUDA_CHECK(cudaHostRegister(
            ptr_,
            size_,
            cudaHostRegisterDefault));
    }

    ~cudaRegister()
    {
        cudaHostUnregister(ptr_);
    }

    cudaRegister(const cudaRegister &) = delete;
    cudaRegister &operator=(const cudaRegister &) = delete;
};