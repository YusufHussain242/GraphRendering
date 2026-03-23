#pragma once
#include "GraphUtils.h"
#include <SFML/Graphics.hpp>
#include <vector>

class GraphEL
{
public:
    std::vector<Vertex> verts;
    std::vector<std::vector<int>> edges;

private:
    void drawEdges(sf::RenderWindow& window);

    void drawVertices(sf::RenderWindow& window, const float vertRadius);

public:
    GraphEL() : verts(0), edges(0) {}

    GraphEL(const int vertCount) : verts(vertCount), edges(vertCount) {}

    GraphEL(const GraphEL& other) : verts(other.verts), edges(other.edges) {}

    GraphEL(GraphEL&& other) noexcept : verts(std::move(other.verts)), edges(std::move(other.edges)) {}

    GraphEL& operator=(const GraphEL& other)
    {
        verts = other.verts;
        edges = other.edges;
        return *this;
    }

    GraphEL& operator=(GraphEL&& other) noexcept
    {
        verts = std::move(other.verts);
        edges = std::move(other.edges);
        return *this;
    }

    void draw(sf::RenderWindow& window, const float vertRadius);

    void printStructure(bool printPositions, bool printEdges);

    void frameGraph(float width, float height, float margin);
};