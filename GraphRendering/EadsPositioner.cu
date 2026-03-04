#include "EadsPositioner.h"
#include <vector>
#include <algorithm>

#include <cuda/cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaUtilityFuncs.h"

const int THREADS_PER_BLOCK = 256;
const int THREADS_PER_BLOCK_2D = 32;

__global__
void getForces(float* X, float* Y, float* EDGE_MASK, float* FX, float* FY, int numVerts, float k1, float k2)
{
    int r = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    int i = r * numVerts + c;

    if (r < numVerts && c < numVerts && r != c)
    {
        float dx = X[c] - X[r];
        float dy = Y[c] - Y[r];
        float dist = sqrtf((dx * dx) + (dy * dy));

        float em_force = -k1 / (dist * dist);
        float elastic_force = k2 * dist * EDGE_MASK[i];
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

__device__ __forceinline__ float getValue2D(const float* data, int r, int c, int numRows, int numCols)
{
    if (r < numRows && c < numCols)
        return data[r * numCols + c];
    else
        return 0.0f;
}

__global__ void reduceAddKernel2D(const float* d_data, float* d_result, int numRows, int inNumCols, int outNumCols)
{
    extern __shared__ float s_data[];

    int s_c = threadIdx.x;
    int d_r = threadIdx.y + blockIdx.y * blockDim.y;
    int d_c = threadIdx.x + blockIdx.x * blockDim.x * 2;

    s_data[s_c] = getValue2D(d_data, d_r, d_c, numRows, inNumCols) + getValue2D(d_data, d_r, d_c + blockDim.x, numRows, inNumCols);

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        __syncthreads();
        if (s_c < offset)
            s_data[s_c] += s_data[s_c + offset];
    }

    if (s_c == 0 && d_r < numRows)
        d_result[d_r * outNumCols + blockIdx.x] = s_data[0];
}

__global__
void vecAdd(float *A, float *B, float *out, int numElems)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < numElems)
        out[i] = A[i] + B[i];
}

void EadsPositioner::positionVertices(Graph& graph)
{
    const int NUM_VERTS = graph.verts.size();

    // Initialize host memory
    float* h_X;
    float* h_Y;
    float* h_EDGE_MASK;
    CUDA_CHECK(cudaMallocHost(&h_X, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMallocHost(&h_Y, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMallocHost(&h_EDGE_MASK, sizeof(float) * NUM_VERTS * NUM_VERTS));

    for (int i = 0; i < NUM_VERTS; i++)
    {
        h_X[i] = graph.verts[i].position.x;
        h_Y[i] = graph.verts[i].position.y;
    }

    for (int u = 0; u < NUM_VERTS; u++)
        for (int v = 0; v < NUM_VERTS; v++)
            h_EDGE_MASK[u * NUM_VERTS + v] = graph.edges[u][v] ? 1.f : 0.f;

    // Initialize device memory
    float* d_X;
    float* d_Y;
    float* d_EDGE_MASK;
    float* d_FX;
    float* d_FY;
    CUDA_CHECK(cudaMalloc(&d_X, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_Y, sizeof(float) * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_EDGE_MASK, sizeof(float) * NUM_VERTS * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_FX, sizeof(float) * NUM_VERTS * NUM_VERTS));
    CUDA_CHECK(cudaMalloc(&d_FY, sizeof(float) * NUM_VERTS * NUM_VERTS));

    // Copy host memory to device memory
    CUDA_CHECK(cudaMemcpy(d_X, h_X, sizeof(float) * NUM_VERTS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, sizeof(float) * NUM_VERTS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_EDGE_MASK, h_EDGE_MASK, sizeof(float) * NUM_VERTS * NUM_VERTS, cudaMemcpyHostToDevice));

    // These are intermidiary buffers used for reduction.
    dim3 reduceBlock(THREADS_PER_BLOCK, 1);
    dim3 reduceGrid(cuda::ceil_div(cuda::ceil_div(NUM_VERTS, 2), reduceBlock.x), NUM_VERTS);
    float* d_DX1;
    float* d_DX2;
    float* d_DY1;
    float* d_DY2;

    CUDA_CHECK(cudaMalloc(&d_DX1, sizeof(float) * NUM_VERTS * reduceGrid.x));
    CUDA_CHECK(cudaMalloc(&d_DX2, sizeof(float) * NUM_VERTS * reduceGrid.x));
    CUDA_CHECK(cudaMalloc(&d_DY1, sizeof(float) * NUM_VERTS * reduceGrid.x));
    CUDA_CHECK(cudaMalloc(&d_DY2, sizeof(float) * NUM_VERTS * reduceGrid.x));

    dim3 forcesBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
    dim3 forcesGrid(NUM_VERTS / forcesBlock.x, NUM_VERTS / forcesBlock.y);
    for (int iter = 0; iter < iters; iter++)
    {
        getForces<<<forcesGrid, forcesBlock>>>(d_X, d_Y, d_EDGE_MASK, d_FX, d_FY, NUM_VERTS, k1, k2);
        
        // The below process will produce a matrix d_DX1 of shape (1, NUM_VERTS) which is identical in
        // memory to a matrix of shape (NUM_VERTS, 1) when using modulo indexing. Hence, d_DX1 can be used
        // as if it were a vector of size NUM_VERTS.
        reduceAddKernel2D<<<reduceGrid, reduceBlock, reduceBlock.x * sizeof(float)>>>(d_FX, d_DX1, NUM_VERTS, NUM_VERTS, reduceGrid.x);
        int curNumCols = reduceGrid.x;
        while (curNumCols > 1)
        {
            dim3 curReduceGrid(cuda::ceil_div((curNumCols + 1) / 2, reduceBlock.x), NUM_VERTS);
            reduceAddKernel2D<<<curReduceGrid, reduceBlock, reduceBlock.x * sizeof(float)>>>(d_DX1, d_DX2, NUM_VERTS, curNumCols, curReduceGrid.x);
            curNumCols = curReduceGrid.x;
            std::swap(d_DX1, d_DX2);
        }

        reduceAddKernel2D<<<reduceGrid, reduceBlock, reduceBlock.x * sizeof(float)>>>(d_FY, d_DY1, NUM_VERTS, NUM_VERTS, reduceGrid.x);
        curNumCols = reduceGrid.x;
        while (curNumCols > 1)
        {
            dim3 curReduceGrid(cuda::ceil_div((curNumCols + 1) / 2, reduceBlock.x), NUM_VERTS);
            reduceAddKernel2D<<<curReduceGrid, reduceBlock, reduceBlock.x * sizeof(float)>>>(d_DY1, d_DY2, NUM_VERTS, curNumCols, curReduceGrid.x);
            curNumCols = curReduceGrid.x;
            std::swap(d_DY1, d_DY2);
        }

        int vecAddThreads = THREADS_PER_BLOCK;
        int vecAddBlocks = cuda::ceil_div(NUM_VERTS, THREADS_PER_BLOCK);
        vecAdd<<<vecAddBlocks, vecAddThreads>>>(d_X, d_DX1, d_X, NUM_VERTS);
        vecAdd<<<vecAddBlocks, vecAddThreads>>>(d_Y, d_DY1, d_Y, NUM_VERTS);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_X, d_X, sizeof(float) * NUM_VERTS, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, sizeof(float) * NUM_VERTS, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_VERTS; i++)
    {
        graph.verts[i].position.x = h_X[i];
        graph.verts[i].position.y = h_Y[i];
    }

    CUDA_CHECK(cudaFree(d_DX1));
    CUDA_CHECK(cudaFree(d_DX2));
    CUDA_CHECK(cudaFree(d_DY1));
    CUDA_CHECK(cudaFree(d_DY2));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_EDGE_MASK));
    CUDA_CHECK(cudaFree(d_FX));
    CUDA_CHECK(cudaFree(d_FY));

    CUDA_CHECK(cudaFreeHost(h_X));
    CUDA_CHECK(cudaFreeHost(h_Y));
    CUDA_CHECK(cudaFreeHost(h_EDGE_MASK));
}

std::string EadsPositioner::getConfigStr()
{
	std::string res = "";
    res += "EADS\n";
	res += "Iters: " + std::to_string(iters) + "\n";
	res += "K1: " + std::to_string(k1) + "\n";
	res += "K2: " + std::to_string(k2) + "\n";
	return res;
}