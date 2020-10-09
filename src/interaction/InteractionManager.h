//
// Created by rutger on 7/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_INTERACTION_MANAGER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_INTERACTION_MANAGER_H

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <iostream>
#include <limits.h>
#include <stdexcept>
#include <map>
#include <SFML/Window/Event.hpp>
#include "KeyPressObserver.h"

class InteractionManager
{
private:
    std::map<sf::Keyboard::Key, KeyPressObserver*> observers;

public:
    void HandleInteraction(const sf::Event& windowEvent)
    {
        if(windowEvent.type == sf::Event::KeyPressed)
        {
            try
            {
                observers.at(windowEvent.key.code)->KeyPressed(windowEvent.key.code);
            }
            catch (const std::out_of_range&)
            {
                //Key is not handled
            }
        }
    }

    void AddObserver(const sf::Keyboard::Key& key, KeyPressObserver* observer)
    {
        observers[key] = observer;
    }
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_INTERACTION_MANAGER_H
