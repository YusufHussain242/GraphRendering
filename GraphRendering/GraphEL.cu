#include "GraphEL.h"

#include <iostream>

void GraphEL::drawEdges(sf::RenderWindow& window)
{ 
    for (int u = 0; u < edges.size(); u++)
        for (int v : edges[u])
            drawLine(window, verts[u].position, verts[v].position);
}

void GraphEL::drawVertices(sf::RenderWindow& window, const float vertRadius)
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

void GraphEL::draw(sf::RenderWindow& window, const float vertRadius)
{
    drawEdges(window);
    drawVertices(window, vertRadius);
}

void GraphEL::printStructure(bool printPositions = true, bool printEdges = true)
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
        for (int u = 0; u < edges.size(); u++)
        {
            std::cout << u << ": ";
            for (int v : edges[u])
                std::cout << v << " ";
            std::cout << std::endl;
        }

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