#include "EadsPositioner2.h"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda/cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaUtilityFuncs.h"

/*
* Optimization Ideas:
* - Put k1 and k2 in constant memory?
* - Have two loops, one for before u, and one for after u.
* - Configure block sizes
* - Put edgeMatrix in constant memory
* - Investigate compiler settings
*/

const int THREADS_PER_BLOCK = 256;
const int TILE_SIZE = 1 * THREADS_PER_BLOCK;

__global__ 
void eadsKernel(float *x_new, float *y_new, float *x_old, float *y_old, int *edgeMatrix, float k1, float k2, int numVerts)
{
    __shared__ float x_tile[TILE_SIZE];
    __shared__ float y_tile[TILE_SIZE];

    const int u = threadIdx.x + blockDim.x * blockIdx.x;
    
    // All __syncthreads calls must be hit by all threads so we cannot early return
    bool active = u < numVerts;

    float acc_x = x_old[u];
    float acc_y = y_old[u];
    float x_old_u = x_old[u];
    float y_old_u = y_old[u];
    int edgeMask = 0;

    for (int tile = 0; tile < numVerts; tile += TILE_SIZE)
    {
        if (active)
        {
            // IMPORTANT: Assumes threads per block evenly divides tile size
            const int width = TILE_SIZE / THREADS_PER_BLOCK;
            for (int i = 0; i < width; i++)
            {
                int idx = threadIdx.x * width + i;
                if (tile + idx < numVerts)
                {
                    x_tile[idx] = x_old[tile + idx];
                    y_tile[idx] = y_old[tile + idx];
                }
            }
        }

        __syncthreads();

        if (active)
        {
            for (int v = 0; tile + v < numVerts && v < TILE_SIZE; v += 32)
            {
                edgeMask = edgeMatrix[u * ((numVerts + 31) >> 5) + ((tile + v) >> 5)];
                for (int bit = 0; tile + v + bit < numVerts && bit < 32; bit++)
                {
                    if (u == tile + v + bit)
                        continue;

                    float dx = x_tile[v + bit] - x_old_u;
                    float dy = y_tile[v + bit] - y_old_u;

                    float dist2 = dx * dx + dy * dy;
                    float invDist = rsqrtf(dist2);

                    float em_force = -k1 * invDist * invDist;
                    float elastic_force = k2 * dist2 * invDist * ((edgeMask >> bit) & 1);
                    float total_force = em_force + elastic_force;

                    acc_x += dx * total_force * invDist;
                    acc_y += dy * total_force * invDist;
                }
            }
        }

        __syncthreads();
    }

    if (active)
    {
        x_new[u] = acc_x;
        y_new[u] = acc_y;
    }
}

void EadsPositioner2::positionVertices(Graph& graph)
{
    const int NUM_VERTS = graph.verts.size();

    const auto startLoading = std::chrono::high_resolution_clock::now();

    // Initialize host memory
    float* h_x;
    float* h_y;
    int* h_edgeMatrix;
    CUDA_CHECK(cudaMallocHost(&h_x, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMallocHost(&h_y, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMallocHost(&h_edgeMatrix, sizeof(int) * NUM_VERTS * cuda::ceil_div(NUM_VERTS, 32)));
    memset(h_edgeMatrix, 0, sizeof(int) * NUM_VERTS * cuda::ceil_div(NUM_VERTS, 32));

    for (int i = 0; i < NUM_VERTS; i++)
    {
        h_x[i] = graph.verts[i].position.x;
        h_y[i] = graph.verts[i].position.y;
    }

    for (int u = 0; u < NUM_VERTS; u++)
        for (int v = 0; v < NUM_VERTS; v++)
            h_edgeMatrix[u * (NUM_VERTS + 31) / 32 + v / 32] |= graph.edges[u][v] << (v % 32);

    // Initialize device memory
    float* d_x1;
    float* d_x2;
    float* d_y1;
    float* d_y2;
    int* d_edgeMatrix;
    CUDA_CHECK(cudaMalloc(&d_x1, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_x2, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_y1, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_y2, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_edgeMatrix, sizeof(int) * NUM_VERTS * cuda::ceil_div(NUM_VERTS, 32)));

    // Copy host memory to device memory
    CUDA_CHECK(cudaMemcpy(d_x2, h_x, sizeof(float) * NUM_VERTS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y2, h_y, sizeof(float) * NUM_VERTS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edgeMatrix, h_edgeMatrix, sizeof(int) * NUM_VERTS * cuda::ceil_div(NUM_VERTS, 32), cudaMemcpyHostToDevice));

    const auto endLoading = std::chrono::high_resolution_clock::now();

    // Main kernel loop
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = cuda::ceil_div(NUM_VERTS, threadsPerBlock);
    for (int iter = 0; iter < iters; iter++)
    {
        eadsKernel<<<blocks, threadsPerBlock>>>(d_x1, d_y1, d_x2, d_y2, d_edgeMatrix, k1, k2, NUM_VERTS);
        std::swap(d_x1, d_x2);
        std::swap(d_y1, d_y2);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const auto endMainLoop = std::chrono::high_resolution_clock::now();
    std::cout << "DATA LOADING TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(endLoading - startLoading) << std::endl;
    std::cout << "MAIN LOOP TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(endMainLoop - endLoading) << std::endl;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_x, d_x1, sizeof(float) * NUM_VERTS, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y, d_y1, sizeof(float) * NUM_VERTS, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_VERTS; i++)
    {
        graph.verts[i].position.x = h_x[i];
        graph.verts[i].position.y = h_y[i];
    }

    // Free memory
    CUDA_CHECK(cudaFree(d_x1));
    CUDA_CHECK(cudaFree(d_x2));
    CUDA_CHECK(cudaFree(d_y1));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaFree(d_edgeMatrix));

    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));
    CUDA_CHECK(cudaFreeHost(h_edgeMatrix));
}

std::string EadsPositioner2::getConfigStr()
{
    std::string res = "";
    res += "EADS\n";
    res += "Iters: " + std::to_string(iters) + "\n";
    res += "K1: " + std::to_string(k1) + "\n";
    res += "K2: " + std::to_string(k2) + "\n";
    return res;
}