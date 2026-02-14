#pragma once
#include <SFML/Graphics.hpp>
#include <set>

// TODO: Need to get rid of this GV namespace

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
        Graph() : verts(0), edges(0) {}

        Graph(const int vertCount) : verts(vertCount), edges(vertCount, std::vector(vertCount, false)) {}

        Graph(const Graph &other) : verts(other.verts), edges(other.edges) {}

        Graph(Graph&& other) noexcept : verts(std::move(other.verts)), edges(std::move(other.edges)) {}

        Graph& operator=(const Graph& other)
        {
            verts = other.verts;
            edges = other.edges;
            return *this;
        }

        Graph& operator=(Graph&& other) noexcept
        {
            verts = std::move(other.verts);
            edges = std::move(other.edges);
            return *this;
        }

        void draw(sf::RenderWindow& window, const float vertRadius);

        void printStructure(bool printPositions, bool printEdges);
    };
}