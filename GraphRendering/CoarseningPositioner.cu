#include "CoarseningPositioner.h"

#include <queue>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cuda/cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaUtilityFuncs.h"

const int THREADS_PER_BLOCK = 256;

std::vector<std::set<int>> CoarseningPositioner::createFiltration(const GraphEL& graph)
{
	int radius = 1;
	std::vector<std::set<int>> res;
	
	std::set<int> vertPool;
	for (int i = 0; i < graph.verts.size(); i++)
		vertPool.insert(i);

	res.push_back(vertPool);

	while (res.empty() || res[res.size() - 1].size() > 1)
	{
		res.push_back(std::set<int>());
		std::vector<bool> visited(graph.verts.size(), false);

		while (!vertPool.empty())
		{
			std::queue<int> q;

			int source = *vertPool.begin();
			vertPool.erase(source);

			res[res.size() - 1].insert(source);
			q.push(source);
			visited[source] = true;

			int dist = 1;
			while (!q.empty() && dist <= radius)
			{
				int curSize = q.size();
				for (int t = 0; t < curSize; t++)
				{
					int cur = q.front();
					q.pop();

					for (int other : graph.edges[cur])
					{
						if (visited[other])
							continue;

						q.push(other);
						visited[other] = true;
						if (vertPool.contains(other))
							vertPool.erase(other);
					}
				}

				dist++;
			}
		}

		radius <<= 1;
		vertPool = res[res.size() - 1];
	}

	return res;
}

std::vector<int> getVertexDepths(const GraphEL& graph, const std::vector<std::set<int>>& filtration)
{
	std::vector<int> res(graph.verts.size(), 0);

	for (int layer = 0; layer < filtration.size(); layer++)
	{
		for (int vert : filtration[layer])
			res[vert] = std::max(res[vert], layer);
	}

	return res;
}

std::vector<std::vector<std::vector<std::pair<int, int>>>> CoarseningPositioner::findNeighbourhoods(const GraphEL& graph, const std::vector<std::set<int>>& filtration)
{
	int edgeCount = 0;
	for (std::vector<int> edgeList : graph.edges)
		edgeCount += edgeList.size();

	std::vector<int> vertexDepths = getVertexDepths(graph, filtration);

	std::vector<std::vector<std::vector<std::pair<int, int>>>> res(graph.verts.size());
	for (int v = 0; v < res.size(); v++)
		res[v] = std::vector<std::vector<std::pair<int, int>>>(vertexDepths[v] + 1, std::vector<std::pair<int, int>>());

	for (int layer = filtration.size() - 1; layer >= 0; layer--)
	{
		std::vector<int> visitedBFS(graph.verts.size(), -1);

		for (int vertex : filtration[layer])
		{
			std::queue<int> q;
			q.push(vertex);
			visitedBFS[vertex] = vertex;

			int dist = 0;
			while (!q.empty() && res[vertex][layer].size() < neighbourhoodSize)
			{
				int curSize = q.size();
				for (int t = 0; t < curSize; t++)
				{
					int cur = q.front();
					q.pop();

					if (dist > 0 && filtration[layer].contains(cur))
						res[vertex][layer].emplace_back(cur, dist);

					if (res[vertex][layer].size() >= neighbourhoodSize)
						break;

					for (int other : graph.edges[cur])
					{
						if (visitedBFS[other] == vertex)
							continue;

						visitedBFS[other] = vertex;
						q.push(other);
					}
				}
				dist++;
			}
		}
	}

	return res;
}

std::vector<std::vector<std::vector<std::pair<int,int>>>> CoarseningPositioner::fastFindNeighbourhoods(const GraphEL& graph, const std::vector<std::set<int>>& filtration)
{
	int edgeCount = 0;
	for (std::vector<int> edgeList : graph.edges)
		edgeCount += edgeList.size();

	std::vector<int> vertexDepths = getVertexDepths(graph, filtration);

	std::vector<std::vector<std::vector<std::pair<int, int>>>> res(graph.verts.size());
	for (int v = 0; v < res.size(); v++)
		res[v] = std::vector<std::vector<std::pair<int, int>>>(vertexDepths[v] + 1, std::vector<std::pair<int, int>>());

	for (int layer = 0; layer < filtration.size(); layer++)
	{
		std::vector<int> visited(graph.verts.size(), -1);
		std::vector<int> dists(graph.verts.size(), INT_MAX);

		for (int vertex : filtration[layer])
		{
			std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
			
			pq.push({ 0, vertex });
			dists[vertex] = 0;
			visited[vertex] = vertex;

			while (!pq.empty() && res[vertex][layer].size() < neighbourhoodSize)
			{
				const auto [dist, cur] = pq.top();
				pq.pop();

				if (dists[cur] < dist)
					continue;

				if (dist > 0 && filtration[layer].contains(cur))
					res[vertex][layer].emplace_back(cur, dist);

				if (res[vertex][layer].size() >= neighbourhoodSize)
					break;

				if (layer > 0)
				{
					for (auto [other, otherDist] : res[cur][layer - 1])
					{
						if (otherDist > (1 << layer) - 1)
							break;

						if (visited[other] != vertex || dist + otherDist < dists[other])
						{
							visited[other] = vertex;
							dists[other] = dist + otherDist;
							pq.push({ dist + otherDist, other });
						}
					}
				}
				else
				{
					for (int other : graph.edges[cur])
					{
						if (visited[other] != vertex || dist + 1 < dists[other])
						{
							visited[other] = vertex;
							dists[other] = dist + 1;
							pq.push({ dist + 1, other });
						}
					}
				}
			}
		}
	}
	
	return res;
}

std::vector<int> CoarseningPositioner::findParentNodes(const GraphEL& graph, const std::vector<std::set<int>>& filtration)
{
	std::vector<int> res(graph.verts.size(), -1);

	std::vector<bool> visited(graph.verts.size(), false);

	for (int layer = filtration.size() - 1; layer >= 0; layer--)
	{
		std::queue<int> q;
		std::vector<int> parentMap(graph.verts.size(), -1);

		for (int vert : filtration[layer])
		{
			q.push(vert);
			visited[vert] = true;
			parentMap[vert] = vert;
		}

		while (!q.empty())
		{
			int cur = q.front();
			q.pop();

			if (!visited[cur])
				res[cur] = parentMap[cur];

			for (int other : graph.edges[cur])
			{
				if (parentMap[other] != -1)
					continue;

				q.push(other);
				parentMap[other] = parentMap[cur];
			}
		}
	}

	return res;
}

__global__
void placeVertsKernel(
	float* x, float* y,
	float* randOffsetX, float* randOffsetY, 
	bool* placed, 
	int* parents, 
	int* filtration, 
	int filtrationSize, 
	float centerX, float centerY
)
{
	const int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= filtrationSize)
		return;

	int vert = filtration[i];

	if (placed[vert])
		return;
	placed[vert] = true;

	if (parents[vert] >= 0)
	{
		x[vert] = x[parents[vert]] + randOffsetX[vert];
		y[vert] = y[parents[vert]] + randOffsetY[vert];
	}
	else
	{
		x[vert] = centerX + randOffsetX[vert];
		y[vert] = centerY + randOffsetY[vert];
	}
}

__global__
void multiLevelKamadaKawaiKernel(
	float* x_new, float* y_new, float* x_old, float* y_old, 
	int* filtration, 
	int* neighbours, 
	int* dists, 
	int layer, 
	int filtrationSize, 
	int neighbourhoodSize, 
	float springStrength, 
	float edgeLength
)
{
	const int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= filtrationSize)
		return;

	int vert = filtration[i];

	x_new[vert] = x_old[vert];
	y_new[vert] = y_old[vert];

	for (int j = i * neighbourhoodSize; j < (i + 1) * neighbourhoodSize; j++)
	{
		int other = neighbours[j];
		if (other == -1)
			continue;

		int avgDist = 1 << layer;
		float normalizedDist = dists[j] / avgDist;

		float k = springStrength / (normalizedDist * normalizedDist);
		float l = edgeLength * dists[j];
		float xDiff = x_old[vert] - x_old[other];
		float yDiff = y_old[vert] - y_old[other];
		float d = sqrtf(xDiff * xDiff + yDiff * yDiff);

		x_new[vert] -= k * (xDiff) * (1 - l / d);
		y_new[vert] -= k * (yDiff) * (1 - l / d);
	}
}

void CoarseningPositioner::positionVerticesGPU(GraphEL& graph)
{
	auto start = std::chrono::high_resolution_clock::now();
	const auto filtration = createFiltration(graph);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "FILTRATION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";

	start = std::chrono::high_resolution_clock::now();
	const auto neighbourhoods = findNeighbourhoods(graph, filtration);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "NEIGHBOURHOOD TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";

	start = std::chrono::high_resolution_clock::now();
	const auto parents = findParentNodes(graph, filtration);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "PARENT TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";

	start = std::chrono::high_resolution_clock::now();

	// Initialize host memory
	float* h_x;
	float* h_y;
	CUDA_CHECK(cudaMallocHost(&h_x, sizeof(float) * graph.verts.size()));
	CUDA_CHECK(cudaMallocHost(&h_y, sizeof(float) * graph.verts.size()));
	
	for (int i = 0; i < graph.verts.size(); i++)
	{
		h_x[i] = graph.verts[i].position.x;
		h_y[i] = graph.verts[i].position.y;
	}

	std::vector<int> filtrationInds;
	std::vector<int> filtrationSizes;
	filtrationInds.push_back(0);
	for (int layer = 0; layer < filtration.size(); layer++)
	{
		filtrationSizes.push_back(filtration[layer].size());
		if (layer < filtration.size() - 1)
			filtrationInds.push_back(filtrationInds[filtrationInds.size() - 1] + filtration[layer].size());
	}

	int* h_filtration;
	int totalFiltrationLen = filtrationInds[filtrationInds.size() - 1] + filtrationSizes[filtrationSizes.size() - 1];
	CUDA_CHECK(cudaMallocHost(&h_filtration, sizeof(int) * totalFiltrationLen));

	int ind = 0;
	for (int layer = 0; layer < filtration.size(); layer++)
	{
		for (int vert : filtration[layer])
		{
			h_filtration[ind] = vert;
			ind++;
		}
	}

	int* h_neighbours;
	int* h_dists;
	CUDA_CHECK(cudaMallocHost(&h_neighbours, sizeof(int) * totalFiltrationLen * neighbourhoodSize));
	CUDA_CHECK(cudaMallocHost(&h_dists, sizeof(int) * totalFiltrationLen * neighbourhoodSize));
	memset(h_neighbours, -1, sizeof(int) * totalFiltrationLen * neighbourhoodSize);

	for (int layer = 0; layer < filtrationInds.size(); layer++)
	{
		for (int i = filtrationInds[layer]; i < filtrationInds[layer] + filtrationSizes[layer]; i++)
		{
			int vert = h_filtration[i];
			for (int j = 0; j < neighbourhoods[vert][layer].size(); j++)
			{
				h_neighbours[i * neighbourhoodSize + j] = neighbourhoods[vert][layer][j].first;
				h_dists[i * neighbourhoodSize + j] = neighbourhoods[vert][layer][j].second;
			}
		}
	}

	bool* h_placed;
	CUDA_CHECK(cudaMallocHost(&h_placed, sizeof(bool) * graph.verts.size()));
	memset(h_placed, false, sizeof(bool) * graph.verts.size());

	int* h_parents;
	CUDA_CHECK(cudaMallocHost(&h_parents, sizeof(int) * graph.verts.size()));
	for (int i = 0; i < parents.size(); i++)
		h_parents[i] = parents[i];

	float* h_randOffsetX;
	float* h_randOffsetY;
	CUDA_CHECK(cudaMallocHost(&h_randOffsetX, sizeof(float) * graph.verts.size()));
	CUDA_CHECK(cudaMallocHost(&h_randOffsetY, sizeof(float) * graph.verts.size()));

	std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> posDistribution(-randRange, randRange);
	for (int i = 0; i < graph.verts.size(); i++)
	{
		h_randOffsetX[i] = posDistribution(rng);
		h_randOffsetY[i] = posDistribution(rng);
	}

	// Initialize device memory
	float* d_x1;
	float* d_x2;
	float* d_y1;
	float* d_y2;
	int* d_filtration;
	int* d_neighbours;
	int* d_dists;
	bool* d_placed;
	int* d_parents;
	float* d_randOffsetX;
	float* d_randOffsetY;
	CUDA_CHECK(cudaMalloc(&d_x1, sizeof(float) * graph.verts.size()));
	CUDA_CHECK(cudaMalloc(&d_x2, sizeof(float) * graph.verts.size()));
	CUDA_CHECK(cudaMalloc(&d_y1, sizeof(float) * graph.verts.size()));
	CUDA_CHECK(cudaMalloc(&d_y2, sizeof(float) * graph.verts.size()));
	CUDA_CHECK(cudaMalloc(&d_filtration, sizeof(int) * totalFiltrationLen));
	CUDA_CHECK(cudaMalloc(&d_neighbours, sizeof(int) * totalFiltrationLen * neighbourhoodSize));
	CUDA_CHECK(cudaMalloc(&d_dists, sizeof(int) * totalFiltrationLen * neighbourhoodSize));
	CUDA_CHECK(cudaMalloc(&d_placed, sizeof(bool) * graph.verts.size()));
	CUDA_CHECK(cudaMalloc(&d_parents, sizeof(int) * graph.verts.size()));
	CUDA_CHECK(cudaMalloc(&d_randOffsetX, sizeof(float) * graph.verts.size()));
	CUDA_CHECK(cudaMalloc(&d_randOffsetY, sizeof(float) * graph.verts.size()));

	// Copy host memory to device memory
	CUDA_CHECK(cudaMemcpy(d_x2, h_x, sizeof(float) * graph.verts.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_y2, h_y, sizeof(float) * graph.verts.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_filtration, h_filtration, sizeof(int) * totalFiltrationLen, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_neighbours, h_neighbours, sizeof(int) * totalFiltrationLen * neighbourhoodSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_dists, h_dists, sizeof(int) * totalFiltrationLen * neighbourhoodSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_placed, h_placed, sizeof(bool) * graph.verts.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_parents, h_parents, sizeof(int) * graph.verts.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_randOffsetX, h_randOffsetX, sizeof(float) * graph.verts.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_randOffsetY, h_randOffsetY, sizeof(float) * graph.verts.size(), cudaMemcpyHostToDevice));

	// Main loop
	for (int layer = filtrationInds.size() - 1; layer >= 0; layer--)
	{
		int threadsPerBlock = THREADS_PER_BLOCK;
		int blocks = cuda::ceil_div(filtrationSizes[layer], threadsPerBlock);

		placeVertsKernel<<<blocks, threadsPerBlock>>>(
			d_x2, d_y2,
			d_randOffsetX, d_randOffsetY,
			d_placed,
			d_parents,
			d_filtration + filtrationInds[layer],
			filtrationSizes[layer],
			centerCoords.x, centerCoords.y
			);

		for (int iter = 0; iter < iters; iter++)
		{
			multiLevelKamadaKawaiKernel<<<blocks, threadsPerBlock>>>(
				d_x1, d_y1, d_x2, d_y2,
				d_filtration + filtrationInds[layer],
				d_neighbours + filtrationInds[layer] * neighbourhoodSize,
				d_dists + filtrationInds[layer] * neighbourhoodSize,
				layer,
				filtrationSizes[layer],
				neighbourhoodSize,
				springStrength,
				edgeLength
				);

				std::swap(d_x1, d_x2);
				std::swap(d_y1, d_y2);
		}
	}

	// Copy results back to host
	CUDA_CHECK(cudaMemcpy(h_x, d_x2, sizeof(float) * graph.verts.size(), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_y, d_y2, sizeof(float) * graph.verts.size(), cudaMemcpyDeviceToHost));

	for (int i = 0; i < graph.verts.size(); i++)
	{
		graph.verts[i].position.x = h_x[i];
		graph.verts[i].position.y = h_y[i];
	}

	// Free memory
	CUDA_CHECK(cudaFree(d_x1));
	CUDA_CHECK(cudaFree(d_x2));
	CUDA_CHECK(cudaFree(d_y1));
	CUDA_CHECK(cudaFree(d_y2));
	CUDA_CHECK(cudaFree(d_filtration));
	CUDA_CHECK(cudaFree(d_neighbours));
	CUDA_CHECK(cudaFree(d_dists));
	CUDA_CHECK(cudaFree(d_placed));

	CUDA_CHECK(cudaFreeHost(h_x));
	CUDA_CHECK(cudaFreeHost(h_y));
	CUDA_CHECK(cudaFreeHost(h_filtration));
	CUDA_CHECK(cudaFreeHost(h_neighbours));
	CUDA_CHECK(cudaFreeHost(h_dists));
	CUDA_CHECK(cudaFreeHost(h_placed));

	end = std::chrono::high_resolution_clock::now();

	std::cout << "SIMULATION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";
}

void CoarseningPositioner::positionVertices(GraphEL& graph)
{
	auto start = std::chrono::high_resolution_clock::now();
	const auto filtration = createFiltration(graph);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "FILTRATION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";

	start = std::chrono::high_resolution_clock::now();
	const auto neighbourhoods = findNeighbourhoods(graph, filtration);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "NEIGHBOURHOOD TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";

	start = std::chrono::high_resolution_clock::now();
	const auto parents = findParentNodes(graph, filtration);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "PARENT TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";

	start = std::chrono::high_resolution_clock::now();

	std::vector<bool> placed(graph.verts.size(), false);
	std::vector<float> dx(graph.verts.size(), 0.f);
	std::vector <float> dy(graph.verts.size(), 0.f);

	std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> posDistribution(-randRange, randRange);

	for (int layer = filtration.size() - 1; layer >= 0; layer--)
	{
		for (int vert : filtration[layer])
		{
			if (placed[vert])
				continue;

			if (parents[vert] >= 0)
			{
				graph.verts[vert].position.x = graph.verts[parents[vert]].position.x + posDistribution(rng);
				graph.verts[vert].position.y = graph.verts[parents[vert]].position.y + posDistribution(rng);
			}
			else
				graph.verts[vert].position = centerCoords + sf::Vector2f(posDistribution(rng), posDistribution(rng));

			placed[vert] = true;
		}

		for (int t = 0; t < iters; t++)
		{
			for (int vert : filtration[layer])
			{
				for (const auto [other, dist] : neighbourhoods[vert][layer])
				{
					int avgDist = 1 << layer;
					float normalizedDist = dist / avgDist;

					float k = springStrength / (normalizedDist * normalizedDist);
					float l = edgeLength * dist;
					float xDiff = graph.verts[vert].position.x - graph.verts[other].position.x;
					float yDiff = graph.verts[vert].position.y - graph.verts[other].position.y;
					float d = sqrtf(xDiff * xDiff + yDiff * yDiff);

					dx[vert] -= k * (xDiff) * (1 - l / d);
					dy[vert] -= k * (yDiff) * (1 - l / d);
				}
			}

			for (int vert : filtration[layer])
			{
				graph.verts[vert].position.x += dx[vert];
				graph.verts[vert].position.y += dy[vert];

				dx[vert] = 0;
				dy[vert] = 0;
			}
		}
	}

	end = std::chrono::high_resolution_clock::now();

	std::cout << "SIMULATION TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";
}