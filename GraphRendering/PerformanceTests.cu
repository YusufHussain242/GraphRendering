#include "PerformanceTests.h"
#include "kernal.h"
#include "Graph.h"
#include "GraphGenerators.h"

#include <chrono>
#include <string>
#include <fstream>

// Takes in a list of vertex and edge counts and randomly generates graphs with
// that number of verts and edges. Then applies the graph positioning algorithm
// to each graph and records performance, logging to a file. If no file path is
// specified, logging will not occur.
//
// TODO: Can refactor this to be a higher order function which takes in a 
// graph positioning algorithm, and tests it.
std::vector<std::chrono::duration<double>> perfTestEads(const std::vector<std::pair<int, int>> &vertAndEdgeCounts, std::string filePath = "")
{
    // The range for where the vertices are generated shouldn't change performance,
    // so it can be set to an abitrary value.
    const float RANGE = 1000.f;

    std::vector<std::chrono::duration<double>> results;
    for (const auto& [vertCount, edgeCount] : vertAndEdgeCounts)
    {
        GV::Graph graph = randomGraph(vertCount, edgeCount, RANGE);
        const auto start = std::chrono::high_resolution_clock::now();
        applyEads(graph, 1000, 1000.f, 0.001f);
        const auto end = std::chrono::high_resolution_clock::now();
        results.push_back(end - start);
    }

    if (filePath != "")
    {
        std::fstream stream(filePath, std::ios::out);
        for (const auto res : results)
            stream << res << "\n";
        stream.close();
    }

    return results;
}

// Measures performance of positioning algorithm at different numbers of vertices,
// starting at lower and increasing by step to upper. The number of edges in the
// graph will be a multiple of the number of verts, determined by edgeRatio.
std::vector<std::chrono::duration<double>> linearPerfTestEads(int lower, int upper, int step, float edgeRatio, std::string filePath = "")
{
    std::vector<std::pair<int, int>> vertAndEdgeCounts;
    for (int i = lower; i <= upper; i += step)
        vertAndEdgeCounts.push_back({ i, static_cast<int>(i * edgeRatio) });

    return perfTestEads(vertAndEdgeCounts, filePath);
}