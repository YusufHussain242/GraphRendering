#include "Graph.h"

#include <iostream>

using namespace GV;

void Graph::drawLine(sf::RenderWindow& window, sf::Vector2f begin, sf::Vector2f end)
{
    std::vector line = { sf::Vertex{begin}, sf::Vertex{end} };
    window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
}

void Graph::drawEdges(sf::RenderWindow& window)
{
    for (int i = 0; i < edges.size(); i++)
        for (int j = 0; j < i; j++)
            if (edges[i][j])
                drawLine(window, verts[i].position, verts[j].position);
}

void Graph::drawVertices(sf::RenderWindow& window, const float vertRadius)
{
    for (Vertex& vert : verts)
    {
        sf::CircleShape circle(vertRadius);
        circle.setPointCount(100);
        circle.setFillColor(sf::Color(255, 0, 0));
        circle.setPosition(vert.position);
        circle.setOrigin({ vertRadius, vertRadius });
        window.draw(circle);
    }
}

void Graph::draw(sf::RenderWindow& window, const float vertRadius)
{
    drawEdges(window);
    drawVertices(window, vertRadius);
}

void Graph::printStructure(bool printPositions = true, bool printEdges = true)
{
    std::cout << "Num verts: " << verts.size() << std::endl;

    if (printPositions)
    {
        std::cout << "Positions:" << std::endl;
        for (int i = 0; i < verts.size(); i++)
            std::cout << i << ": " << verts[i].position.x << ", " << verts[i].position.y << std::endl;
        std::cout << std::endl;
    }

    if (printEdges)
    {
        std::cout << "Edges:" << std::endl;
        for (int i = 0; i < verts.size(); i++)
        {
            std::cout << i << ": ";
            for (int j = 0; j < verts.size(); j++)
                if (edges[i][j])
                    std::cout << j << " ";
            std::cout << std::endl;
        }
    }
}