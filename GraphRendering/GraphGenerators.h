#pragma once
#include "Graph.h"
#include "GraphEL.h"

Graph readGraph();

Graph randomGraph(int vertCount, int edgeCount, float range);

GraphEL randomGraphEL(int vertCount, int edgeCount, float range);