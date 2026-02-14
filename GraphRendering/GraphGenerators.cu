#include "GraphGenerators.h"

#include <iostream>
#include <random>
#include <chrono>
#include <vector>

GV::Graph readGraph()
{
    int vertCount;
    std::cout << "Enter number of vertices" << std::endl;
    std::cin >> vertCount;

    int edgeCount;
    std::cout << "Enter number of edges" << std::endl;
    std::cin >> edgeCount;

    GV::Graph graph(vertCount);
    std::cout << "Enter " << vertCount << " vertex positions\n";
    for (int i = 0; i < graph.verts.size(); i++)
    {
        float x, y;
        std::cin >> x >> y;
        graph.verts[i].position = sf::Vector2f(x, y);
    }

    std::cout << "Enter " << edgeCount << " edges\n";
    for (int i = 0; i < edgeCount; i++)
    {
        int u, v;
        std::cin >> u >> v;
        graph.edges[u][v] = true;
        graph.edges[v][u] = true;
    }

    return graph;
}

GV::Graph randomGraph(int vertCount, int edgeCount, float range)
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

    std::uniform_real_distribution<float> posDistribution(0.0f, range);
    for (GV::Vertex& vert : graph.verts)
        vert.position = { posDistribution(rng), posDistribution(rng) };

    return graph;
}
