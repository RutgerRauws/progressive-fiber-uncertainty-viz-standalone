//
// Created by rutger on 11/18/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CAMERA_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CAMERA_H


#include <libs/glm/mat4x4.hpp>

class Camera
{
    public:
        const float FOV = 45.0f;

        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
        glm::mat4 projectionMatrix;

        glm::vec3 cameraPos;
        glm::vec3 cameraFront;
        glm::vec3 cameraUp;

        float yaw = -90.0f; //make sure camera points towards the negative z-axis by default
        float pitch = 0.0f;

        Camera(const glm::vec3& cameraPos, const glm::vec3& cameraFront, const glm::vec3& cameraUp);

        void LookAt(const glm::vec3& position, const glm::vec3& center, const glm::vec3& up);

        void ResizeWindow(int w, int h);

};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CAMERA_H
