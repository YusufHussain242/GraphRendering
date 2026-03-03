#pragma once
#include "Graph.h"

template <typename T>
concept IGraphPositioner = requires(T v, Graph &graph) {
    { v.positionVertices(graph) } -> std::same_as<void>;
    { v.getConfigStr() } -> std::convertible_to<std::string>;
};