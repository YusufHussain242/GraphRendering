#include <SFML/Graphics.hpp>
#include "Graph.h"
#include "GraphEL.h"
#include "GraphGenerators.h"
#include "EadsPositioner.h"
#include "EadsPositioner2.h"
#include "MultiLevelCPUPositioner.h"
#include "CoarseningPositioner.h"
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
    GraphEL graph = serpinskyGraphEL(9, std::min(WINDOW_WIDTH, WINDOW_HEIGHT));

    /*
    MultiLevelCPUPositioner positioner;
    positioner.iters = 200;
    positioner.clusterNumber = 10;
    positioner.edgeLength = std::min(WINDOW_WIDTH, WINDOW_HEIGHT) / 40;
    positioner.springStrength = 0.2;
    positioner.centerCoords = { WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2 };
    
    positioner.positionVerticesKK(graph);
    */

    CoarseningPositioner positioner;
    positioner.iters = 200;
    positioner.neighbourhoodSize = 20;
    positioner.edgeLength = std::min(WINDOW_WIDTH, WINDOW_HEIGHT) / 700;
    positioner.springStrength = 0.2f;
    positioner.randRange = 10.f;
    positioner.centerCoords = { WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2 };
    

    positioner.positionVertices(graph);

    /*
    CoarseningPositioner positioner2;
    positioner2.iters = 200;
    positioner2.neighbourMult = 3;
    positioner2.edgeLength = std::min(WINDOW_WIDTH, WINDOW_HEIGHT) / 40;
    positioner2.springStrength = 0.2f;
    positioner2.randRange = 10.f;
    positioner2.centerCoords = { WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2 };

    const auto filtration = positioner2.createFiltration(graph);
    const auto neighbourhoods = positioner2.findNeighbourhoods(graph, filtration);

    int vert = 0;
    int maxCount = 12;
    int count = 0;
    int layer = 2;
    for (; vert < neighbourhoods.size(); vert++)
    {
        if (neighbourhoods[vert].size() > layer)
            count++;

        if (count >= maxCount)
            break;
    }

    graph.verts[vert].color = { 0, 0, 255 };
    for (auto [other, dist] : neighbourhoods[vert][layer])
        graph.verts[other].color = { 255, 0, 0 };
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
        graph.draw(window, 0);
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