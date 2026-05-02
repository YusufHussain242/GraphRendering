#pragma once
#include "IGraphPositioner.h"

#include <string>

class EadsPositioner
{
public:
	int iters;
	float k1;
	float k2;
	float k3;
	float k4;
	bool timeResults;

	void positionVertices(Graph& graph);
	std::string getConfigStr();
};