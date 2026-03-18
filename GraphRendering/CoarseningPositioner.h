#pragma once
#include "GraphEL.h"

#include <vector>
#include <set>

class CoarseningPositioner
{
private:

public:
	std::vector<std::set<int>> createFiltration(const GraphEL& graph);
	
	std::vector<std::vector<std::pair<int, int>>> findNeighbourhoods(const GraphEL& graph, const std::vector<std::set<int>>& filtration);
	
	void positionVertices(GraphEL& graph);
	std::string getConfigStr();
};