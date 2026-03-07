#pragma once
#include "IGraphPositioner.h"

#include <string>

class EadsPositioner2
{
public:
	int iters;
	float k1;
	float k2;

	void positionVertices(Graph& graph);
	std::string getConfigStr();
};