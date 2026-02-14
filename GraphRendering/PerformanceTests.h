#pragma once
#include <string>
#include <vector>
#include <chrono>

std::vector<std::chrono::duration<double>> perfTestEads(const std::vector<std::pair<int, int>>& vertAndEdgeCounts, std::string filePath);

std::vector<std::chrono::duration<double>> linearPerfTestEads(int lower, int upper, int step, float edgeRatio, std::string filePath);