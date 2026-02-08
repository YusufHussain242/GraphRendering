#include <SFML/Graphics.hpp>

#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include "kernal.h"
#include "Graph.h"

GV::Graph readGraph(int vertCount, int edgeCount)
{
    GV::Graph graph(vertCount);
    std::cout << "Enter " << vertCount << " vertex positions\n";
    for (int i = 0; i < graph.verts.size(); i++)
    {
        float x, y;
        std::cin >> x >> y;
        graph.verts[i].position = sf::Vector2f(x, y);
    }

    std::cout << "Enter " << edgeCount << " Edges\n";
    for (int i = 0; i < edgeCount; i++)
    {
        int u, v;
        std::cin >> u >> v;
        graph.edges[u][v] = true;
        graph.edges[u][v] = true;
    }

    return graph;
}

GV::Graph generateGraph(int vertCount, int edgeCount)
{
    GV::Graph graph(vertCount);

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> vertDistribution(0, vertCount - 1);

    int curEdges = 0;
    while (curEdges < edgeCount)
    {
        int vert1 = vertDistribution(rng);
        int vert2 = vertDistribution(rng);
        if (!graph.edges[vert1][vert2] && !graph.edges[vert2][vert1])
        {
            graph.edges[vert1][vert2] = true;
            graph.edges[vert2][vert1] = true;
            curEdges++;
        }
    }

    std::uniform_real_distribution<float> posDistribution(0.0f, 1000.f);
    for (GV::Vertex& vert : graph.verts)
        vert.position = { posDistribution(rng), posDistribution(rng) };

    return graph;
}

int main()
{
    int numVerts;
    std::cout << "Enter node count" << std::endl;
    std::cin >> numVerts;
    
    int numEdges;
    std::cout << "Enter edge count" << std::endl;
    std::cin >> numEdges;

    GV::Graph graph = generateGraph(numVerts, numEdges);

    std::cout << "Before:\n";
    graph.printStructure(true, true);
    
    applyEads(graph, 1000, 1000.f, 0.001f);

    std::cout << "After\n";
    graph.printStructure(true, true);

    sf::RenderWindow window(sf::VideoMode({ 1000, 1000 }), "Graph Renderer");

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear();
        graph.draw(window, 15.f);
        window.display();
    }
}