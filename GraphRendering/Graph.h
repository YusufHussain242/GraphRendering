#pragma once

#include <SFML/Graphics.hpp>
#include <set>

namespace GV
{
    struct Vertex
    {
        sf::Vector2f position;
    };

    class Graph
    {
    public:
        std::vector<Vertex> verts;
        std::vector<std::vector<bool>> edges;

    private:
        void drawLine(sf::RenderWindow& window, sf::Vector2f begin, sf::Vector2f end);

        void drawEdges(sf::RenderWindow& window);

        void drawVertices(sf::RenderWindow& window, const float vertRadius);

    public:
        Graph(const int vertCount) : verts(vertCount), edges(vertCount, std::vector(vertCount, false)) {}

        void draw(sf::RenderWindow& window, const float vertRadius);

        void printStructure(bool printPositions, bool printEdges);
    };
}