#include <stdio.h>
#include <cuda_runtime_api.h>

template <class FunctorType>
__global__ void call_device_functor(FunctorType device_functor)
{
    device_functor();
}

class TestFunctor
{
public:
    __device__ __host__ TestFunctor(int x) : x(x)
    {
        printf("constructor called\n");
        cudaMalloc((void **)&x, sizeof(int));
    }

    __host__ TestFunctor(const TestFunctor &other)
    {
        printf("copy constructor called\n");
        x = other.x;
        y = other.y;
    }

    __device__ __host__ ~TestFunctor()
    {
        cudaFree(y);
        printf("destructor called with y: %d\n", y);
    }

    __device__ __host__ void operator()()
    {
        printf("%d\n", x);
    }

private:
    int x;
    int *y;
};

int main()
{
    TestFunctor test(15);
    printf("line1 and 2\n");
    call_device_functor<<<3, 3>>>(test);
    printf("line2 and 3\n");
    cudaDeviceSynchronize();
}