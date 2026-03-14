#include "MultiLevelCPUPositioner.h"

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <algorithm>
#include <random>

Clustering MultiLevelCPUPositioner::createClusterHierarchy(GraphEL& graph, int k)
{
	Clustering clustering;
	clustering.center = 0;
	clustering.size = graph.verts.size(); // This is technically not true if the graph is not fully connected.

	std::vector<std::vector<int>> clusterMaps = { std::vector<int>(graph.verts.size(), 0) };

	std::vector<int> visitedFindCenters(graph.verts.size(), -1);
	std::vector<int> visitedClusterMap(graph.verts.size(), -1);
	std::vector<int> visitedDistMatrix(graph.verts.size(), -1);

	std::queue<Clustering*> q;
	q.push(&clustering);

	while (!q.empty())
	{
		int numClusters = q.size();
		std::vector<int> nextClusterMap = clusterMaps[clusterMaps.size() - 1];
		for (int i = 0; i < numClusters; i++)
		{
			Clustering* cur = q.front();
			q.pop();

			findKCenters(graph, *cur, clusterMaps[clusterMaps.size() - 1], visitedFindCenters, k);
			fillClusterMap(graph, *cur, clusterMaps[clusterMaps.size() - 1], nextClusterMap, visitedClusterMap);
			fillDistMatrix(graph, *cur, clusterMaps[clusterMaps.size() - 1], visitedDistMatrix);

			for (Clustering& cluster : cur->clusters)
				if (cluster.size > 1)
					q.push(&cluster);
		}

		clusterMaps.emplace_back(std::move(nextClusterMap));

		std::fill(visitedFindCenters.begin(), visitedFindCenters.end(), -1);
		std::fill(visitedClusterMap.begin(), visitedClusterMap.end(), -1);
		std::fill(visitedDistMatrix.begin(), visitedDistMatrix.end(), -1);
	}

	std::map<int, sf::Color> colorMap;
	std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<int> dist(0, 255);
	for (int i : clusterMaps[1])
	{
		if (!colorMap.contains(i))
			colorMap[i] = sf::Color(dist(rng), dist(rng), dist(rng), 255);
	}

	for (int i = 0; i < graph.verts.size(); i++)
		graph.verts[i].color = colorMap[clusterMaps[1][i]];

	return clustering;
}

void MultiLevelCPUPositioner::findKCenters(GraphEL& graph, Clustering& clustering, std::vector<int>& clusterMap, std::vector<int>& visited, int k)
{
	clustering.clusters.emplace_back(Clustering());
	clustering.clusters[0].center = clustering.center;

	std::queue<int> q;
	for (int i = 1; i < k; i++)
	{
		for (Clustering &cluster : clustering.clusters)
		{
			q.push(cluster.center);
			visited[cluster.center] = i;
		}

		int nextCenter = -1;
		while (!q.empty())
		{
			int vert = q.front();
			q.pop();

			for (int other : graph.edges[vert])
			{
				if ((visited[other] == i) || (clusterMap[other] != clustering.center))
					continue;
				
				q.push(other);
				visited[other] = i;
				nextCenter = other;
			}
		}

		if (nextCenter != -1)
		{
			clustering.clusters.emplace_back(Clustering());
			clustering.clusters[clustering.clusters.size() - 1].center = nextCenter;
		}
	}
}

void MultiLevelCPUPositioner::fillClusterMap(GraphEL& graph, Clustering& clustering, std::vector<int>& curClusterMap, std::vector<int>& nextClusterMap, std::vector<int>& visited)
{
	int k = clustering.clusters.size();

	int clusterSize = 0;
	std::queue<int> q;
	std::map<int, Clustering*> centerToClustering; // Maybe there's a way to get rid of this map?
	for (Clustering& cluster : clustering.clusters)
	{
		q.push(cluster.center);
		visited[cluster.center] = 0;
		nextClusterMap[cluster.center] = cluster.center;
		centerToClustering[cluster.center] = &cluster;
		cluster.size++;
	}

	while (!q.empty())
	{
		int vert = q.front();
		q.pop();

		for (int other : graph.edges[vert])
		{
			if ((visited[other] == 0) || (curClusterMap[other] != clustering.center))
				continue;
			q.push(other);
			visited[other] = 0;
			nextClusterMap[other] = nextClusterMap[vert];
			centerToClustering[nextClusterMap[vert]]->size++;
		}
	}
}

// Might be a good idea to have edge weights be max(d(center[i], center[j]), radius[i] + radius[j]))
// although this won't work well asymetrical clusters with bad centers, need to approximate long distance
// forces from parents instead.
// 
// Should be able to remove this function and do this in findCenters function
// Could remove last cluster check here.
void MultiLevelCPUPositioner::fillDistMatrix(GraphEL& graph, Clustering& clustering, std::vector<int>& clusterMap, std::vector<int>& visited)
{
	int k = clustering.clusters.size();
	
	clustering.distMatrix.resize(k, std::vector<int>());
	for (int i = 0; i < k; i++)
		clustering.distMatrix[i].resize(k, INT_MAX);

	std::queue<int> q;
	for (int i = 0; i < k; i++)
	{
		q.push(clustering.clusters[i].center);
		visited[clustering.clusters[i].center] = i;

		int distance = 0;
		while (!q.empty())
		{
			int curQueueSize = q.size();
			for (int t = 0; t < curQueueSize; t++)
			{
				int vert = q.front();
				q.pop();

				// Could optimize this by having a global isCenter bool vector
				for (int j = 0; j < k; j++)
					if (vert == clustering.clusters[j].center)
						clustering.distMatrix[i][j] = distance;

				for (int other : graph.edges[vert])
				{
					if ((visited[other] == i) || (clusterMap[other] != clustering.center))
						continue;
					q.push(other);
					visited[other] = i;
				}
			}
			distance++;
		}
	}
}

std::vector<std::vector<float>> floydWarshall(GraphEL& graph)
{
	std::vector<std::vector<float>> dist(graph.verts.size(), std::vector<float>(graph.verts.size(), INFINITY));

	for (int u = 0; u < graph.verts.size(); u++)
	{
		dist[u][u] = 0.f;
		for (int v : graph.edges[u])
			dist[u][v] = 1.f;
	}

	for (int k = 0; k < graph.verts.size(); k++)
		for (int i = 0; i < graph.verts.size(); i++)
			for (int j = 0; j < graph.verts.size(); j++)
				if (dist[i][k] != INFINITY && dist[k][j] != INFINITY)
					dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);

	return dist;
}

void MultiLevelCPUPositioner::positionVerticesEads(GraphEL& graph)
{
	std::vector<std::vector<float>> dist = floydWarshall(graph);

	std::vector<float> dx(graph.verts.size());
	std::vector<float> dy(graph.verts.size());

	for (int i = 0; i < iters; i++)
	{
		std::fill(dx.begin(), dx.end(), 0);
		std::fill(dy.begin(), dy.end(), 0);
		for (int u = 0; u < graph.verts.size(); u++)
		{
			for (int v = 0; v < graph.verts.size(); v++)
			{
				if (u == v)
					continue;

				float k = springStrength / (dist[u][v] * dist[u][v]);
				float l = edgeLength * dist[u][v];
				float xDiff = graph.verts[u].position.x - graph.verts[v].position.x;
				float yDiff = graph.verts[u].position.y - graph.verts[v].position.y;
				float d = sqrtf(xDiff * xDiff + yDiff * yDiff);
				
				dx[u] -= k * (xDiff) * (1 - l / d);
				dy[u] -= k * (yDiff) * (1 - l / d);
			}
		}

		for (int u = 0; u < graph.verts.size(); u++)
		{
			graph.verts[u].position.x += dx[u];
			graph.verts[u].position.y += dy[u];
		}
	}
}

void MultiLevelCPUPositioner::positionClusters(Clustering& clustering)
{
	std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> posDistribution(-50.f, 50.f);
	for (Clustering& cluster : clustering.clusters)
	{
		// Keep the cluster center locked in place as an anchor.
		if (cluster.center == clustering.center)
			cluster.position = clustering.position;
		else
			cluster.position = clustering.position + sf::Vector2f(posDistribution(rng), posDistribution(rng));
	}

	std::vector<float> dx(clustering.clusters.size());
	std::vector<float> dy(clustering.clusters.size());

	for (int i = 0; i < iters; i++)
	{
		std::fill(dx.begin(), dx.end(), 0);
		std::fill(dy.begin(), dy.end(), 0);
		for (int u = 0; u < clustering.clusters.size(); u++)
		{
			if (clustering.clusters[u].center == clustering.center)
				continue;

			for (int v = 0; v < clustering.clusters.size(); v++)
			{
				if (u == v)
					continue;

				float k = clustering.clusters[v].size * springStrength / (clustering.distMatrix[u][v] * clustering.distMatrix[u][v]);
				float l = edgeLength * clustering.distMatrix[u][v];
				float xDiff = clustering.clusters[u].position.x - clustering.clusters[v].position.x;
				float yDiff = clustering.clusters[u].position.y - clustering.clusters[v].position.y;
				float d = sqrtf(xDiff * xDiff + yDiff * yDiff);

				dx[u] -= k * (xDiff) * (1 - l / d);
				dy[u] -= k * (yDiff) * (1 - l / d);
			}
		}

		for (int u = 0; u < clustering.clusters.size(); u++)
		{
			clustering.clusters[u].position.x += dx[u];
			clustering.clusters[u].position.y += dy[u];
		}
	}
}

void MultiLevelCPUPositioner::positionSubClusters(Clustering& clustering)
{
	std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> posDistribution(-50.f, 50.f);
	for (Clustering& cluster : clustering.clusters)
	{
		for (Clustering& subCluster : cluster.clusters)
		{
			// Keep the cluster centers locked in place as an anchor.
			if (subCluster.center == cluster.center)
				subCluster.position = cluster.position;
			else
				subCluster.position = cluster.position + sf::Vector2f(posDistribution(rng), posDistribution(rng));
		}
	}

	for (int i = 0; i < clustering.clusters.size(); i++)
	{
		Clustering& cluster = clustering.clusters[i];
		if (cluster.clusters.size() == 0)
			continue;

		std::vector<float> dx(cluster.clusters.size());
		std::vector<float> dy(cluster.clusters.size());

		for (int t = 0; t < iters; t++)
		{
			std::fill(dx.begin(), dx.end(), 0);
			std::fill(dy.begin(), dy.end(), 0);

			for (int u = 0; u < cluster.clusters.size(); u++)
			{
				if (cluster.clusters[u].center == cluster.center)
					continue;

				for (int j = 0; j < clustering.clusters.size(); j++)
				{
					if (i == j)
						continue;

					Clustering& otherCluster = clustering.clusters[j];

					// Might need to dampen spring strength here to account for inaccuracy
					float k = otherCluster.size * springStrength / (clustering.distMatrix[i][j] * clustering.distMatrix[i][j]);
					float l = edgeLength * clustering.distMatrix[i][j];
					float xDiff = cluster.clusters[u].position.x - otherCluster.position.x;
					float yDiff = cluster.clusters[u].position.y - otherCluster.position.y;
					float d = sqrtf(xDiff * xDiff + yDiff * yDiff);

					dx[u] -= k * (xDiff) * (1 - l / d);
					dy[u] -= k * (yDiff) * (1 - l / d);
				}

				for (int v = 0; v < cluster.clusters.size(); v++)
				{
					if (u == v)
						continue;

					float k = cluster.clusters[v].size * springStrength / (cluster.distMatrix[u][v] * cluster.distMatrix[u][v]);
					float l = edgeLength * cluster.distMatrix[u][v];
					float xDiff = cluster.clusters[u].position.x - cluster.clusters[v].position.x;
					float yDiff = cluster.clusters[u].position.y - cluster.clusters[v].position.y;
					float d = sqrtf(xDiff * xDiff + yDiff * yDiff);

					dx[u] -= k * (xDiff) * (1 - l / d);
					dy[u] -= k * (yDiff) * (1 - l / d);
				}
			}

			for (int u = 0; u < cluster.clusters.size(); u++)
			{
				cluster.clusters[u].position.x += dx[u];
				cluster.clusters[u].position.y += dy[u];
			}
		}
	}
}

void MultiLevelCPUPositioner::positionVertices(GraphEL& graph)
{
	Clustering clustering = createClusterHierarchy(graph, clusterNumber);
	clustering.position = centerCoords;
	positionClusters(clustering);

	std::queue<Clustering*> q;
	q.push(&clustering);
	while (!q.empty())
	{
		Clustering* cur = q.front();
		q.pop();

		positionSubClusters(*cur);

		for (Clustering& cluster : cur->clusters)
		{
			if (cluster.size > 1)
				q.push(&cluster);
			else
				graph.verts[cluster.center].position = cluster.position;
		}
	}
}