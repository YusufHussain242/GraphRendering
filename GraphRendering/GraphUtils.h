#pragma once
#include <SFML/Graphics.hpp>

struct Vertex
{
    sf::Vector2f position = sf::Vector2f(0.f, 0.f);
    sf::Color color = sf::Color(255, 255, 255, 255);
};

void drawLine(sf::RenderWindow& window, sf::Vector2f begin, sf::Vector2f end);