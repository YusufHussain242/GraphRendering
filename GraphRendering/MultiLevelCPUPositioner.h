#pragma once
#include "IGraphPositioner.h"
#include "GraphEL.h"

#include <SFML/Graphics.hpp>
#include <string>

struct Clustering
{
	int center;
	int size = 0;
	sf::Vector2f position;
	std::vector<Clustering> clusters;
	std::vector<std::vector<int>> distMatrix;
};

class MultiLevelCPUPositioner
{
public:
	int iters;
	int clusterNumber;
	float edgeLength;
	float springStrength;
	sf::Vector2f centerCoords;

	void positionVertices(GraphEL& graph);
	
	void positionClusters(Clustering& clustering);

	void positionSubClusters(Clustering& clustering);

	void positionVerticesKK(GraphEL& graph);

	std::string getConfigStr();

	Clustering createClusterHierarchy(GraphEL& graph, int k);

private:
	void findKCenters(GraphEL& graph, Clustering& clustering, std::vector<int>& clusterMap, std::vector<int>& visited, int k);

	void fillClusterMap(GraphEL& graph, Clustering& clustering, std::vector<int>& curClusterMap, std::vector<int>& nextClusterMap, std::vector<int>& visited);

	void fillDistMatrix(GraphEL& graph, Clustering& clustering, std::vector<int>& clusterMap, std::vector<int>& visited);
};