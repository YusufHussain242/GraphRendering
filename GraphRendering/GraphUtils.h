#pragma once
#include <SFML/Graphics.hpp>

struct Vertex
{
    sf::Vector2f position;
};

void drawLine(sf::RenderWindow& window, sf::Vector2f begin, sf::Vector2f end);