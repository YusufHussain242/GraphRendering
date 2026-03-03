#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

// Takes in a list of vertex and edge counts and randomly generates graphs with
// that number of verts and edges. Then applies the graph positioning algorithm
// to each graph and records performance, logging to a file. If no file path is
// specified, logging will not occur.
//
// TODO: Can refactor this to be a higher order function which takes in a 
// graph positioning algorithm, and tests it.
template <IGraphPositioner T>
std::vector<std::chrono::duration<double>> perfTestRandom(T& positioner, const std::vector<std::pair<int, int>>& vertAndEdgeCounts, std::string filePath = "")
{
    // The range for where the vertices are generated shouldn't change performance,
    // so it can be set to an abitrary value.
    const float RANGE = 1000.f;

    std::vector<std::chrono::duration<double>> results;
    for (const auto& [vertCount, edgeCount] : vertAndEdgeCounts)
    {
        Graph graph = randomGraph(vertCount, edgeCount, RANGE);
        const auto start = std::chrono::high_resolution_clock::now();
        positioner.positionVertices(graph);
        const auto end = std::chrono::high_resolution_clock::now();
        results.push_back(end - start);
    }

    if (filePath != "")
    {
        std::fstream stream(filePath, std::ios::out);
        stream << "CONFIGURATION:\n" << positioner.getConfigStr() << "\n";

        for (int i = 0; i < vertAndEdgeCounts.size(); i++)
        {
            stream << "TEST: " << i << "\n";
            stream << "VERTS: " << vertAndEdgeCounts[i].first << "\n";
            stream << "EDGES: " << vertAndEdgeCounts[i].second << "\n";
            stream << "TIME: " << results[i] << "\n";
            stream << "\n";
        }

        stream.close();
    }

    return results;
}

// Measures performance of positioning algorithm at different numbers of vertices,
// starting at lower and increasing by step to upper. The number of edges in the
// graph will be a multiple of the number of verts, determined by edgeRatio.
template <IGraphPositioner T>
std::vector<std::chrono::duration<double>> linearPerfTestRandom(T positioner, int lower, int upper, int step, float edgeRatio, std::string filePath = "")
{
    std::vector<std::pair<int, int>> vertAndEdgeCounts;
    for (int i = lower; i <= upper; i += step)
        vertAndEdgeCounts.push_back({ i, static_cast<int>(i * edgeRatio) });

    return perfTestRandom(positioner, vertAndEdgeCounts, filePath);
}