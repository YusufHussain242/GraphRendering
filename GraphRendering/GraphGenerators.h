#pragma once
#include "Graph.h"

Graph readGraph();

Graph randomGraph(int vertCount, int edgeCount, float range);

std::vector<std::vector<int>> randomEdgeListGraph(int vertCount, int edgeCount);