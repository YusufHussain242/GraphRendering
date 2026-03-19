#include "CoarseningPositioner.h"

#include <queue>
#include <random>
#include <algorithm>
#include <iostream>

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

// This function can be optimised.
std::vector<std::vector<std::vector<std::pair<int,int>>>> CoarseningPositioner::findNeighbourhoods(const GraphEL& graph, const std::vector<std::set<int>>& filtration)
{
	// We want each the total number of neighbours at each layer to be constant.
	// We want the number of neighbours per vertex to be avg_deg(G) at the finest layer.
	// So the total number of neighbours in the finest layer is |V| * avg_deg(G) = |E|.
	// Hnece, the number of neighers per vertex in layer i is: |E| / |V_i|.

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

		int numNeighbours = neighbourMult * (edgeCount + filtration[layer].size() - 1) / filtration[layer].size();
		for (int vertex : filtration[layer])
		{
			std::queue<int> q;
			q.push(vertex);
			visitedBFS[vertex] = vertex;

			int dist = 0;
			while (!q.empty() && res[vertex][layer].size() < numNeighbours)
			{
				int curSize = q.size();
				for (int t = 0; t < curSize; t++)
				{
					int cur = q.front();
					q.pop();

					if (dist > 0 && filtration[layer].contains(cur))
						res[vertex][layer].emplace_back(cur, dist);
					
					if (res[vertex][layer].size() >= numNeighbours)
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

void CoarseningPositioner::positionVertices(GraphEL& graph)
{
	const auto filtration = createFiltration(graph);
	const auto neighbourhoods = findNeighbourhoods(graph, filtration);
	const auto parents = findParentNodes(graph, filtration);

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

					float k = springStrength / (dist * dist);
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
}