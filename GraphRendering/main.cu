#include <SFML/Graphics.hpp>
#include "kernal.h"
#include "Graph.h"
#include "GraphGenerators.h"
#include "PerformanceTests.h"

#include <iostream>

const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 1000;

void renderGraph()
{
    GV::Graph graph;
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

    linearPerfTestEads(lower, upper, step, edgeRatio, "C:/PerformanceResults/" + testName);
}

int main()
{
    while (true)
    {
        std::cout << "CHOSE OPTION:" << std::endl;
        std::cout << "1. Render Graph" << std::endl;
        std::cout << "2. Performance Tests" << std::endl;

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
        }
    }
}