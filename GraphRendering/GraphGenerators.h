#pragma once
#include "Graph.h"
#include "GraphEL.h"

Graph readGraph();

Graph randomGraph(int vertCount, int edgeCount, float range);

GraphEL randomGraphEL(int vertCount, int edgeCount, float range);

GraphEL serpinskyGraphEL(int triNumber, float range);

Graph latticeGraph(int width, int height, float range);

Graph EdgeListToAdjacencyMatrix(GraphEL graphEL);