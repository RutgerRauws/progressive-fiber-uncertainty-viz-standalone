//
// Created by rutger on 11/18/20.
//

#include <libs/glm/ext.hpp>
#include "Camera.h"

Camera::Camera(const glm::vec3& cameraPos, const glm::vec3& cameraFront, const glm::vec3& cameraUp)
    : cameraPos(cameraPos), cameraFront(cameraFront), cameraUp(cameraUp)
{
    modelMatrix = glm::mat4(1.0f);
    viewMatrix = glm::mat4(1.0f);
    viewMatrix = glm::translate(viewMatrix, glm::vec3(0, 0, -3));
    projectionMatrix = glm::mat4(0); //It's set by ResizeWindow() anyways

    LookAt(cameraPos, cameraFront, cameraUp);
}

void Camera::ResizeWindow(int w, int h)
{
    projectionMatrix = glm::perspective(
            glm::radians(FOV),
            (float)w / (float)h,
            0.1f,
            10000.0f
    );
}

void Camera::LookAt(const glm::vec3& position, const glm::vec3& center, const glm::vec3& up)
{
    cameraPos = position;
    cameraFront = center;

    viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
}
