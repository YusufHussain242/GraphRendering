#include "MultiLevelCPUPositioner.h"

#include <iostream>
#include <vector>
#include <queue>
#include <map>

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