#include "kernal.h"
#include <iostream>
#include <vector>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

__global__
void getForces(float* X, float* Y, float* EDGE_MASK, float* FX, float* FY, int numVerts, float k1, float k2)
{
    int r = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    int i = r * numVerts + c;

    if (r < numVerts && c < numVerts && r != c)
    {
        float dx = X[r] - X[c];
        float dy = Y[r] - Y[c];
        float dist = sqrtf((dx * dx) + (dy * dy));

        float em_force = k1 / (dist * dist);
        float elastic_force = -k2 * dist * EDGE_MASK[i];
        float total_force = em_force + elastic_force;

        if (dist > 0.0001f)
        {
            FX[i] = dx * total_force / dist;
            FY[i] = dy * total_force / dist;
        }
        else
        {
            FX[i] = 0;
            FY[i] = 0;
        }
    }
}

void applyEads(GV::Graph &graph, const int iters, const float k1, const float k2)
{
    const int NUM_VERTS = graph.verts.size();
    float* X;
    float* Y;
    float* EDGE_MASK;
    float* FX;
    float* FY;

    CUDA_CHECK(cudaMallocManaged(&X, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMallocManaged(&Y, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMallocManaged(&EDGE_MASK, sizeof(float) * NUM_VERTS * NUM_VERTS));
    CUDA_CHECK(cudaMallocManaged(&FX, sizeof(float) * NUM_VERTS * NUM_VERTS));
    CUDA_CHECK(cudaMallocManaged(&FY, sizeof(float) * NUM_VERTS * NUM_VERTS));

    // We need to zero out the FX, and FY vectors. Not sure if this is working
    // it's causing CUDA runtime errors
    CUDA_CHECK(cudaMemset(&FX, 0, sizeof(float) * NUM_VERTS * NUM_VERTS));
    CUDA_CHECK(cudaMemset(&FY, 0, sizeof(float) * NUM_VERTS * NUM_VERTS));

    for (int i = 0; i < NUM_VERTS; i++)
    {
        X[i] = graph.verts[i].position.x;
        Y[i] = graph.verts[i].position.y;
	}

    for (int u = 0; u < NUM_VERTS; u++)
        for (int v = 0; v < NUM_VERTS; v++)
            EDGE_MASK[u * NUM_VERTS + v] = graph.edges[u][v] ? 1.f : 0.f;

    for (int iter = 0; iter <= iters; iter++)
    {
        int gridSize = (NUM_VERTS + 31) / 32;
        dim3 grid(gridSize, gridSize);
        dim3 block(32, 32);
        getForces<<<grid, block>>>(X, Y, EDGE_MASK, FX, FY, NUM_VERTS, k1, k2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int r = 0; r < NUM_VERTS; r++)
        {
            for (int c = 0; c < NUM_VERTS; c++)
            {
                X[r] += FX[r * NUM_VERTS + c];
                Y[r] += FY[r * NUM_VERTS + c];
            }
        }

        // Need to get rid of this for loop
        /*
        for (int i = 0; i < NUM_VERTS; i++)
        {
            X[i] += thrust::reduce(thrust::device, FX + i * NUM_VERTS, FX + i * (NUM_VERTS + 1));
            Y[i] += thrust::reduce(thrust::device, FY + i * NUM_VERTS, FY + i * (NUM_VERTS + 1));
        }
        */
    }

    for (int i = 0; i < NUM_VERTS; i++)
    {
        graph.verts[i].position.x = X[i];
        graph.verts[i].position.y = Y[i];
    }

    CUDA_CHECK(cudaFree(X));
    CUDA_CHECK(cudaFree(Y));
    CUDA_CHECK(cudaFree(EDGE_MASK));
    CUDA_CHECK(cudaFree(FX));
    CUDA_CHECK(cudaFree(FY));
}