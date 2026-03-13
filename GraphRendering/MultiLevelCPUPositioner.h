#pragma once
#include "IGraphPositioner.h"
#include "GraphEL.h"

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
	void positionVertices(GraphEL& graph);

	std::string getConfigStr();

	Clustering createClusterHierarchy(GraphEL& graph, int k);

private:
	void findKCenters(GraphEL& graph, Clustering& clustering, std::vector<int>& clusterMap, std::vector<int>& visited, int k);

	void fillClusterMap(GraphEL& graph, Clustering& clustering, std::vector<int>& curClusterMap, std::vector<int>& nextClusterMap, std::vector<int>& visited);

	void fillDistMatrix(GraphEL& graph, Clustering& clustering, std::vector<int>& clusterMap, std::vector<int>& visited);
};