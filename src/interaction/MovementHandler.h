//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H


#include "../util/glm/mat4x4.hpp"

class MovementHandler
{
    private:
        const float MOVEMENT_SPEED = 8.0f;
        const float ROTATE_SPEED = 0.8f;

        sf::Window& window;
        sf::Vector2i centerPos;

        glm::mat4& modelMatrix;
        glm::mat4& viewMatrix;
        glm::mat4& projectionMatrix;

        glm::vec3 cameraPos;
        glm::vec3 cameraFront;
        glm::vec3 cameraUp;
        float yaw = -90.0f; //make sure camera points towards the negative z-axis by default
        float pitch = 0.0f;

        sf::Vector2i getMouseDeltaAndReset();

    public:
        MovementHandler(
            sf::Window& window,
            glm::mat4& modelMatrix,
            glm::mat4& viewMatrix,
            glm::mat4& projectionMatrix
        );

        void update();

        void SetCameraPosition(glm::vec3 position);
        void SetCameraFront(glm::vec3 direction);
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H
