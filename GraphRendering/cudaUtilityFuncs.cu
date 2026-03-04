#include "cudaUtilityFuncs.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <cuda/cmath>

__device__ __forceinline__ float getValue(const float* data, int index, int numElements)
{
    if (index < numElements)
        return data[index];
    else
        return 0.0f;
}

__global__ void reduceAddKernel(const float* d_data, float* d_result, int numElements)
{
    extern __shared__ float s_data[];

    int s_i = threadIdx.x;
    int d_i = threadIdx.x + blockIdx.x * blockDim.x * 2;
    s_data[s_i] = getValue(d_data, d_i, numElements) + getValue(d_data, d_i + blockDim.x, numElements);

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        __syncthreads();
        if (s_i < offset)
            s_data[s_i] += s_data[s_i + offset];
    }

    if (s_i == 0)
        d_result[blockIdx.x] = s_data[0];
}

float reduceAdd(std::vector<float> array, bool timeResults)
{
    const int NUM_BYTES = array.size() * sizeof(float);

    float* h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, NUM_BYTES));
    CUDA_CHECK(cudaMemcpy(h_data, array.data(), NUM_BYTES, cudaMemcpyHostToHost));

    float h_result = 0.0f;

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, NUM_BYTES));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, NUM_BYTES, cudaMemcpyHostToDevice));

    int threads = 256;
    // Only need half as many blocks to cover array, since each thread reduces two blocks.
    int blocks = cuda::ceil_div(cuda::ceil_div(array.size(), 2), threads);
    
    float* d_result1;
    float* d_result2;
    CUDA_CHECK(cudaMalloc(&d_result1, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result2, blocks * sizeof(float)));

    const auto start = std::chrono::high_resolution_clock::now();

    reduceAddKernel<<<blocks, threads, threads * sizeof(float)>>>(d_data, d_result1, array.size());
    int curElemCount = blocks;
    while (curElemCount > 1)
    {
        blocks = cuda::ceil_div(cuda::ceil_div(curElemCount, 2), threads);
        reduceAddKernel<<<blocks, threads, threads * sizeof(float)>>>(d_result1, d_result2, curElemCount);
        curElemCount = blocks;
        std::swap(d_result1, d_result2);
    }

    CUDA_CHECK(cudaMemcpy(&h_result, d_result1, sizeof(float), cudaMemcpyDeviceToHost));
    
    const auto stop = std::chrono::high_resolution_clock::now();
    if (timeResults)
        std::cout << "REDUCTION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start) << "\n";

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result1));
    CUDA_CHECK(cudaFree(d_result2));
    CUDA_CHECK(cudaFreeHost(h_data));
    
    return h_result;
}