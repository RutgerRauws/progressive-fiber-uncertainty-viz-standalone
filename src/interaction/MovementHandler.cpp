//
// Created by rutger on 10/8/20.
//

#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Mouse.hpp>
#include <iostream>
#include "MovementHandler.h"
#include "glm/ext.hpp"


MovementHandler::MovementHandler(glm::mat4& modelMatrix,
                                 glm::mat4& viewMatrix,
                                 glm::mat4& projectionMatrix)
    : cameraState(
        modelMatrix, viewMatrix, projectionMatrix,
        CAMERA_POS,
        CAMERA_FRT,
        CAMERA_UP
//        glm::vec3(0, 0, 0),
//        glm::vec3(0, 0, -1),
//        glm::vec3(0, 1, 0)
      )
{
    cameraState.viewMatrix = glm::lookAt(cameraState.cameraPos,
                                         cameraState.cameraPos + cameraState.cameraFront,
                                         cameraState.cameraUp
    );
}


void MovementHandler::update()
{
//    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Left) || sf::Keyboard::isKeyPressed(sf::Keyboard::A))
//    {
//        cameraState.cameraPos -= MOVEMENT_SPEED * glm::normalize(glm::cross(cameraState.cameraFront, cameraState.cameraUp));
//    }
//
//    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::D))
//    {
//        cameraState.cameraPos += MOVEMENT_SPEED * glm::normalize(glm::cross(cameraState.cameraFront, cameraState.cameraUp));
//    }
//
//    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Up) || sf::Keyboard::isKeyPressed(sf::Keyboard::W))
//    {
//        cameraState.cameraPos += MOVEMENT_SPEED * cameraState.cameraFront;
//    }
//
//    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Down) || sf::Keyboard::isKeyPressed(sf::Keyboard::S))
//    {
//        cameraState.cameraPos -= MOVEMENT_SPEED * cameraState.cameraFront;
//    }
}

void MovementHandler::SetCameraPosition(glm::vec3 position)
{
    cameraState.cameraPos = position;
}

void MovementHandler::SetCameraFront(glm::vec3 direction)
{
    cameraState.cameraFront = direction;
}


void MovementHandler::MouseMovement(const glm::ivec2& mouseDelta)
{
    if(mouseDelta.x != 0 || mouseDelta.y != 0)
    {
        cameraState.yaw   -= ROTATE_SPEED * mouseDelta.x;
        cameraState.pitch += ROTATE_SPEED * mouseDelta.y;

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

        cameraState.viewMatrix = glm::lookAt(cameraState.cameraPos,
                                             cameraState.cameraPos + cameraState.cameraFront,
                                             cameraState.cameraUp
        );
    }
}

const CameraState& MovementHandler::GetCameraState()
{
    return cameraState;
}