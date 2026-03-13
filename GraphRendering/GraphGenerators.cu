#include "GraphGenerators.h"

#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <set>

void randomizeVertexPositions(GraphEL& graph, float range)
{
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> posDistribution(0.0f, range);
    for (Vertex& vert : graph.verts)
        vert.position = { posDistribution(rng), posDistribution(rng) };
}

Graph readGraph()
{
    int vertCount;
    std::cout << "Enter number of vertices" << std::endl;
    std::cin >> vertCount;

    int edgeCount;
    std::cout << "Enter number of edges" << std::endl;
    std::cin >> edgeCount;

    Graph graph(vertCount);
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

Graph randomGraph(int vertCount, int edgeCount, float range)
{
    Graph graph(vertCount);

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
    for (Vertex& vert : graph.verts)
        vert.position = { posDistribution(rng), posDistribution(rng) };

    return graph;
}

GraphEL randomGraphEL(int vertCount, int edgeCount, float range)
{
    std::vector<std::set<int>> graphES(vertCount);

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> vertDistribution(0, vertCount - 1);

    int curEdges = 0;
    while (curEdges < edgeCount)
    {
        int vert1 = vertDistribution(rng);
        int vert2 = vertDistribution(rng);
        if (!graphES[vert1].contains(vert2) && !graphES[vert2].contains(vert1))
        {
            graphES[vert1].insert(vert2);
            graphES[vert2].insert(vert1);
            curEdges++;
        }
    }

    GraphEL graphEL(vertCount);
    for (int i = 0; i < vertCount; i++)
        graphEL.edges[i] = std::vector<int>(graphES[i].begin(), graphES[i].end());

    std::uniform_real_distribution<float> posDistribution(0.0f, range);
    for (Vertex& vert : graphEL.verts)
        vert.position = { posDistribution(rng), posDistribution(rng) };

    return graphEL;
}

int fast_expo(int a, int b)
{
    if (b == 0)
        return 1;

    if (b % 2 == 0)
    {
        int temp = fast_expo(a, b / 2);
        return temp * temp;
    }
    else
        return a * fast_expo(a, b - 1);
}

std::tuple<int, int, int> serpinskyHelper(GraphEL &graph, int index, int n)
{
    if (n == 1)
        return std::make_tuple(index, index, index);

    auto [up1, dl1, dr1] = serpinskyHelper(graph, index + 0 * (n / 3), n / 3);
    auto [up2, dl2, dr2] = serpinskyHelper(graph, index + 1 * (n / 3), n / 3);
    auto [up3, dl3, dr3] = serpinskyHelper(graph, index + 2 * (n / 3), n / 3);

    graph.edges[dl1].push_back(up2);
    graph.edges[up2].push_back(dl1);

    graph.edges[dr1].push_back(up3);
    graph.edges[up3].push_back(dr1);

    graph.edges[dr2].push_back(dl3);
    graph.edges[dl3].push_back(dr2);

    return std::make_tuple(up1, dl2, dr3);
}

GraphEL serpinskyGraphEL(int triNumber, float range)
{
    GraphEL graph(fast_expo(3, triNumber));
    serpinskyHelper(graph, 0, graph.verts.size());
    randomizeVertexPositions(graph, range);
    return graph;
}