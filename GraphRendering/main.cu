#include <SFML/Graphics.hpp>
#include "Graph.h"
#include "GraphEL.h"
#include "GraphGenerators.h"
#include "EadsPositioner.h"
#include "EadsPositioner2.h"
#include "MultiLevelCPUPositioner.h"
#include "PerformanceTests.h"
#include "cudaUtilityFuncs.h"

#include <iostream>
#include <random>

const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 1000;

void renderGraph()
{
    Graph graph;
    while (true)
    {
        const int NUM_OPTIONS = 2;
        std::cout << "ENTER GRAPH CONSTRUCTION METHOD:" << std::endl;
        std::cout << "1. MANUAL" << std::endl;
        std::cout << "2. RANDOM" << std::endl;

        int option;
        std::cin >> option;

        switch (option)
        {
        case 1:
            graph = readGraph();
            break;
        case 2:
            int vertCount;
            std::cout << "Enter number of vertices" << std::endl;
            std::cin >> vertCount;

            int edgeCount;
            std::cout << "Enter number of edges" << std::endl;
            std::cin >> edgeCount;

            graph = randomGraph(vertCount, edgeCount, std::min(WINDOW_WIDTH, WINDOW_HEIGHT));
            break;
        }

        if (option > 0 && option <= NUM_OPTIONS)
            break;
    }

    EadsPositioner2 positioner;
    positioner.iters = 1000;
    positioner.k1 = 1000.0f;
    positioner.k2 = 0.001f;
    positioner.positionVertices(graph);

    sf::RenderWindow window(sf::VideoMode({ WINDOW_WIDTH, WINDOW_HEIGHT }), "Graph Renderer");
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

void performanceTests()
{
    std::string testName;
    int lower, upper, step;
    float edgeRatio;

    std::cout << "Enter performance test name:" << std::endl;
    std::cin >> testName;

    std::cout << "Enter lower bound for vertex count:" << std::endl;
    std::cin >> lower;

    std::cout << "Enter upper bound for vertex count:" << std::endl;
    std::cin >> upper;

    std::cout << "Enter step for vertex count:" << std::endl;
    std::cin >> step;

    std::cout << "Enter edge ratio (ratio of edges relative to vertices):" << std::endl;
    std::cin >> edgeRatio;

    EadsPositioner2 positioner;
    positioner.iters = 1000;
    positioner.k1 = 1000.0f;
    positioner.k2 = 0.001f;
    linearPerfTestRandom(positioner, lower, upper, step, edgeRatio, "../PerformanceResults/" + testName);
}

void tempFunc()
{
    GraphEL graph = randomGraphEL(10, 20, std::min(WINDOW_WIDTH, WINDOW_HEIGHT));
    /*
    GraphEL graph(10);
    int k = 1;
    for (int i = 0; i + k < graph.verts.size(); i++)
    {
        for (int j = i + 1; j <= i + k; j++)
        {
            graph.edges[i].push_back(j);
            graph.edges[j].push_back(i);
        }
    }

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> posDistribution(0.0f, std::min(WINDOW_WIDTH, WINDOW_HEIGHT));
    for (Vertex& vert : graph.verts)
        vert.position = { posDistribution(rng), posDistribution(rng) };
    */

    MultiLevelCPUPositioner positioner;
    positioner.edgeLength = std::min(WINDOW_WIDTH, WINDOW_HEIGHT) / 10;
    positioner.springStrength = 0.001;
    positioner.iters = 50;
    
    positioner.positionVertices(graph);

    graph.printStructure(true, true);
    
    /*
    auto start = std::chrono::high_resolution_clock::now();
    Clustering clustering = positioner.createClusterHierarchy(graph, 10);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CLUSTER HIERARCHY TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << "\n";
    */

    sf::RenderWindow window(sf::VideoMode({ WINDOW_WIDTH, WINDOW_HEIGHT }), "Graph Renderer");
    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear();
        graph.draw(window, 10.f);
        window.display();
    }
}

int main()
{
    while (true)
    {
        std::cout << "CHOSE OPTION:" << std::endl;
        std::cout << "1. Render Graph" << std::endl;
        std::cout << "2. Performance Tests" << std::endl;
        std::cout << "3. Temp" << std::endl;

        int option;
        std::cin >> option;
        switch(option)
        {
        case 1:
            renderGraph();
            break;
        case 2:
            performanceTests();
            break;
        case 3:
            tempFunc();
            break;
        }
    }
}