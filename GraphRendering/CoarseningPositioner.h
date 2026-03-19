#pragma once
#include "GraphEL.h"

#include <vector>
#include <set>

class CoarseningPositioner
{
private:
	std::vector<std::set<int>> createFiltration(const GraphEL& graph);

	std::vector<std::vector<std::pair<int, int>>> findNeighbourhoods(const GraphEL& graph, const std::vector<std::set<int>>& filtration);

	std::vector<int> findParentNodes(const GraphEL& graph, const std::vector<std::set<int>>& filtration);

public:
	int iters;
	float randRange;
	float springStrength;
	float edgeLength;
	sf::Vector2f centerCoords;

	void positionVertices(GraphEL& graph);
	
	std::string getConfigStr();
};