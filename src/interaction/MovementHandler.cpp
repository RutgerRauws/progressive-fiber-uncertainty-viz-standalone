//
// Created by rutger on 10/8/20.
//

#include "MovementHandler.h"
#include "glm/ext.hpp"

MovementHandler::MovementHandler(Camera& camera)
    : camera(camera)
{}

//void MovementHandler::update()
//{
//    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Left) || sf::Keyboard::isKeyPressed(sf::Keyboard::A))
//    {
//        camera.cameraPos -= MOVEMENT_SPEED * glm::normalize(glm::cross(camera.cameraFront, camera.cameraUp));
//    }
//
//    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::D))
//    {
//        camera.cameraPos += MOVEMENT_SPEED * glm::normalize(glm::cross(camera.cameraFront, camera.cameraUp));
//    }
//
//    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Up) || sf::Keyboard::isKeyPressed(sf::Keyboard::W))
//    {
//        camera.cameraPos += MOVEMENT_SPEED * camera.cameraFront;
//    }
//
//    if(sf::Keyboard::isKeyPressed(sf::Keyboard::Down) || sf::Keyboard::isKeyPressed(sf::Keyboard::S))
//    {
//        camera.cameraPos -= MOVEMENT_SPEED * camera.cameraFront;
//    }
//}

void MovementHandler::MouseMovement(const glm::ivec2& mouseDelta)
{
    if(mouseDelta.x != 0 || mouseDelta.y != 0)
    {
        camera.yaw   -= ROTATE_SPEED * mouseDelta.x;
        camera.pitch += ROTATE_SPEED * mouseDelta.y;

        if(camera.pitch > 89.0f) {
            camera.pitch = 89.0f;
        }

        if(camera.pitch < -89.0f) {
            camera.pitch = -89.0f;
        }

        glm::vec3 direction;
        direction.x = cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
        direction.y = sin(glm::radians(camera.pitch));
        direction.z = sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));

        camera.LookAt(camera.cameraPos, direction, camera.cameraUp);
    }
}