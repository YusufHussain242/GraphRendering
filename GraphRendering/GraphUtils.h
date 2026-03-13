#pragma once
#include <SFML/Graphics.hpp>

struct Vertex
{
    sf::Vector2f position = sf::Vector2f(0.f, 0.f);
};

void drawLine(sf::RenderWindow& window, sf::Vector2f begin, sf::Vector2f end);