#include "CoarseningPositioner.h"

#include <queue>

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

std::vector<std::vector<std::pair<int,int>>> CoarseningPositioner::findNeighbourhoods(const GraphEL& graph, const std::vector<std::set<int>>& filtration)
{
	// We want each the total number of neighbours at each layer to be constant.
	// We want the number of neighbours per vertex to be avg_deg(G) at the finest layer.
	// So the total number of neighbours in the finest layer is |V| * avg_deg(G) = |E|.
	// Hnece, the number of neighers per vertex in layer i is: |E| / |V_i|.

	int edgeCount = 0;
	for (std::vector<int> edgeList : graph.edges)
		edgeCount += edgeList.size();

	std::vector<std::vector<std::pair<int, int>>> res(graph.verts.size(), std::vector<std::pair<int,int>>());
	std::vector<bool> visited(graph.verts.size(), false);
	std::vector<int> visitedBFS(graph.verts.size(), -1);

	for (int layer = filtration.size() - 1; layer >= 0; layer--)
	{
		int numNeighbours = (edgeCount + filtration[layer].size() - 1) / filtration[layer].size();
		for (int vertex : filtration[layer])
		{
			if (visited[vertex])
				continue;

			visited[vertex] = true;

			std::queue<int> q;
			q.push(vertex);
			visitedBFS[vertex] = vertex;

			int dist = 0;
			while (!q.empty() && res[vertex].size() < numNeighbours)
			{
				int curSize = q.size();
				for (int t = 0; t < curSize; t++)
				{
					int cur = q.front();
					q.pop();

					if (dist > 0 && filtration[layer].contains(cur))
						res[vertex].emplace_back(cur, dist);
					
					if (res[vertex].size() >= numNeighbours)
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