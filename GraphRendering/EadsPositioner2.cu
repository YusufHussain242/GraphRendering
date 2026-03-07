#include "EadsPositioner2.h"

#include <iostream>
#include <algorithm>
#include <cuda/cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaUtilityFuncs.h"

/*
* Optimization Ideas:
* - Store EDGE_MASK booleans using bits
* - Put k1 and k2 in constant memory?
* - Use register cache
*/

const int THREADS_PER_BLOCK = 256;

__global__ 
void eadsKernel(float *x_new, float *y_new, float *x_old, float *y_old, bool *edgeMatrix, float k1, float k2, int numVerts)
{
	int u = threadIdx.x + blockDim.x * blockIdx.x;

	if (u < numVerts)
	{
        x_new[u] = x_old[u];
        y_new[u] = y_old[u];
		for (int v = 0; v < numVerts; v++)
		{
            if (u != v)
            {
                float dx = x_old[v] - x_old[u];
                float dy = y_old[v] - y_old[u];
                float dist = sqrtf((dx * dx) + (dy * dy));

                float em_force = -k1 / (dist * dist);
                float elastic_force = k2 * dist * edgeMatrix[u * numVerts + v];
                float total_force = em_force + elastic_force;

                x_new[u] += dx * total_force / dist;
                y_new[u] += dy * total_force / dist;
            }
		}
	}
}

void EadsPositioner2::positionVertices(Graph& graph)
{
    const int NUM_VERTS = graph.verts.size();

    // Initialize host memory
    float* h_x;
    float* h_y;
    bool* h_edgeMatrix;
    CUDA_CHECK(cudaMallocHost(&h_x, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMallocHost(&h_y, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMallocHost(&h_edgeMatrix, sizeof(bool) * NUM_VERTS * NUM_VERTS));

    for (int i = 0; i < NUM_VERTS; i++)
    {
        h_x[i] = graph.verts[i].position.x;
        h_y[i] = graph.verts[i].position.y;
    }

    for (int u = 0; u < NUM_VERTS; u++)
        for (int v = 0; v < NUM_VERTS; v++)
            h_edgeMatrix[u * NUM_VERTS + v] = graph.edges[u][v];

    // Initialize device memory
    float* d_x1;
    float* d_x2;
    float* d_y1;
    float* d_y2;
    bool* d_edgeMatrix;
    CUDA_CHECK(cudaMalloc(&d_x1, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_x2, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_y1, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_y2, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_edgeMatrix, sizeof(bool) * NUM_VERTS * NUM_VERTS));

    // Copy host memory to device memory
    CUDA_CHECK(cudaMemcpy(d_x2, h_x, sizeof(float) * NUM_VERTS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y2, h_y, sizeof(float) * NUM_VERTS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edgeMatrix, h_edgeMatrix, sizeof(bool) * NUM_VERTS * NUM_VERTS, cudaMemcpyHostToDevice));

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