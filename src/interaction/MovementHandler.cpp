//
// Created by rutger on 10/8/20.
//

#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Mouse.hpp>
#include <iostream>
#include <GL/glew.h>
#include "MovementHandler.h"
#include "../util/glm/ext.hpp"


MovementHandler::MovementHandler(sf::Window& window,
                                 glm::mat4 &modelMatrix,
                                 glm::mat4 &viewMatrix,
                                 glm::mat4 &projectionMatrix)
    : window(window),
      cameraState(
      modelMatrix, viewMatrix, projectionMatrix,
      glm::vec3(0, 0, 0),
      glm::vec3(0, 0, -1),
      glm::vec3(0, 1, 0)
      )
{
    centerPos.x = window.getSize().x / 2;
    centerPos.y = window.getSize().y / 2;
//    window.setMouseCursorGrabbed(true);
//    window.setMouseCursorVisible(false);

//    window.setPosition({-100, -100});

    getMouseDeltaAndReset();
}


void MovementHandler::update()
{
    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Left) || sf::Keyboard::isKeyPressed(sf::Keyboard::A))
    {
        cameraState.cameraPos -= MOVEMENT_SPEED * glm::normalize(glm::cross(cameraState.cameraFront, cameraState.cameraUp));
    }

    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::D))
    {
        cameraState.cameraPos += MOVEMENT_SPEED * glm::normalize(glm::cross(cameraState.cameraFront, cameraState.cameraUp));
    }

    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Up) || sf::Keyboard::isKeyPressed(sf::Keyboard::W))
    {
        cameraState.cameraPos += MOVEMENT_SPEED * cameraState.cameraFront;
    }

    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Down) || sf::Keyboard::isKeyPressed(sf::Keyboard::S))
    {
        cameraState.cameraPos -= MOVEMENT_SPEED * cameraState.cameraFront;
    }

    // -x is left +x is right. -y is up +y is down
    sf::Vector2i mouseDelta = getMouseDeltaAndReset();

    if(mouseDelta.x != 0 || mouseDelta.y != 0)
    {
        cameraState.yaw   += ROTATE_SPEED * mouseDelta.x;
        cameraState.pitch -= ROTATE_SPEED * mouseDelta.y;

        if(cameraState.pitch > 89.0f) {
            cameraState.pitch = 89.0f;
        }

        if(cameraState.pitch < -89.0f) {
            cameraState.pitch = -89.0f;
        }

        glm::vec3 direction;
        direction.x = cos(glm::radians(cameraState.yaw)) * cos(glm::radians(cameraState.pitch));
        direction.y = sin(glm::radians(cameraState.pitch));
        direction.z = sin(glm::radians(cameraState.yaw)) * cos(glm::radians(cameraState.pitch));

        cameraState.cameraFront = glm::normalize(direction);
    }

    cameraState.viewMatrix = glm::lookAt(cameraState.cameraPos,
                                         cameraState.cameraPos + cameraState.cameraFront,
                                         cameraState.cameraUp
    );
//    std::cout << "Pos: " << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << std::endl;
//    std::cout << "Frt: " << cameraFront.x << ", " << cameraFront.y << ", " << cameraFront.z << std::endl;
//    std::cout << "Up: " << cameraUp.x << ", " << cameraUp.y << ", " << cameraUp.z << std::endl;
}

sf::Vector2i MovementHandler::getMouseDeltaAndReset()
{
    if(!window.hasFocus())
    {
        return {0,0};
    }

    sf::Vector2i pos = sf::Mouse::getPosition(window);
    sf::Vector2i delta = pos - centerPos;

    sf::Mouse::setPosition(centerPos, window);

    return delta;
}

void MovementHandler::SetCameraPosition(glm::vec3 position)
{
    cameraState.cameraPos = position;
}

void MovementHandler::SetCameraFront(glm::vec3 direction)
{
    cameraState.cameraFront = direction;
}

const CameraState &MovementHandler::GetCameraState() {
    return cameraState;
}
