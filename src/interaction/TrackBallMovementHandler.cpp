//
// Created by rutger on 18/11/20.
//

#include "TrackBallMovementHandler.h"
#include "glm/ext.hpp"

TrackBallMovementHandler::TrackBallMovementHandler(Camera &camera)
        : vDist(0), theta(0), phi(0),
          MovementHandler(camera)
{
    camera.cameraPos   = glm::vec3(200, 0, 0);
    camera.cameraFront = glm::vec3(-200, 0, 0);
    camera.cameraUp    = glm::vec3(0, 0, 1);

    camera.LookAt(camera.cameraPos, camera.cameraFront, camera.cameraUp);

    glm::vec3 v = camera.cameraPos;

    vDist = glm::length(v);
    theta = std::atan2(v.z / v.x, v.x);
    phi = std::acos(v.z / vDist);
}

void TrackBallMovementHandler::MouseMovement(const glm::ivec2& mouseDelta)
{
    if(mouseDelta.x != 0 || mouseDelta.y != 0)
    {
        // Rotate the camera left and right
        phi -= mouseDelta.x * ROTATE_SPEED;

        // Rotate the camera up and down
        // Prevent the camera from turning upside down (1.5f = approx. Pi / 2)
        theta = glm::clamp(theta - mouseDelta.y * ROTATE_SPEED, -1.5f, 1.5f);

        // Calculate the cartesian coordinates
        camera.cameraPos = getCartesianCoordinates();
        camera.cameraFront = -camera.cameraPos;

        // Make the camera look at the target
        camera.LookAt(camera.cameraPos, camera.cameraFront, camera.cameraUp);
    }
}

void TrackBallMovementHandler::MouseScroll(int delta)
{
    vDist -= (float)delta * ZOOM_SPEED;

    camera.cameraPos = getCartesianCoordinates();
    camera.cameraFront = -camera.cameraPos;

    // Make the camera look at the target
    camera.LookAt(camera.cameraPos, camera.cameraFront, camera.cameraUp);
}

glm::vec3 TrackBallMovementHandler::getCartesianCoordinates() const
{
    return glm::vec3(
        vDist * std::cos(theta) * std::sin(phi),
        vDist * std::cos(theta) * std::cos(phi),
        vDist * std::sin(theta)
    );
}
