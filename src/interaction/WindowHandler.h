//
// Created by rutger on 10/25/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_WINDOW_HANDLER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_WINDOW_HANDLER_H

#include <SFML/Graphics/RenderWindow.hpp>
#include "KeyPressObserver.h"

class WindowHandler : public KeyPressObserver
{
    private:
        sf::RenderWindow& window;

    public:
        explicit WindowHandler(sf::RenderWindow& window) : window(window) {}

        void KeyPressed(const sf::Keyboard::Key& key) override
        {
            window.close();
        }
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_WINDOW_HANDLER_H
