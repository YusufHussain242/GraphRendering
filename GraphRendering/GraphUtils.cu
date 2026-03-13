#include "GraphUtils.h"

void drawLine(sf::RenderWindow& window, sf::Vector2f begin, sf::Vector2f end)
{
    std::vector line = { sf::Vertex{begin}, sf::Vertex{end} };
    window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
}