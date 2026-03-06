#pragma once
#include "IGraphPositioner.h"

#include <string>

class EadsPositioner
{
public:
	int iters;
	float k1;
	float k2;
	bool timeResults;

	void positionVertices(Graph& graph);
	std::string getConfigStr();
};