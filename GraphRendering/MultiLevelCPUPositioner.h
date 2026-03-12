#pragma once
#include "IGraphPositioner.h"

#include <string>

struct Clustering
{
	int center;

	int size = 0;

	std::vector<Clustering> clusters;

	std::vector<std::vector<int>> distMatrix;
};

class MultiLevelCPUPositioner
{
public:
	void positionVertices(Graph& graph);

	std::string getConfigStr();

	Clustering createClusterHierarchy(std::vector<std::vector<int>>& graph, int k);

private:

	void findKCenters(std::vector<std::vector<int>>& graph, Clustering& clustering, std::vector<int>& clusterMap, std::vector<int>& visited, int k);

	void fillClusterMap(std::vector<std::vector<int>>& graph, Clustering& clustering, std::vector<int>& curClusterMap, std::vector<int>& nextClusterMap, std::vector<int>& visited);

	void fillDistMatrix(std::vector<std::vector<int>>& graph, Clustering& clustering, std::vector<int>& clusterMap, std::vector<int>& visited);
};