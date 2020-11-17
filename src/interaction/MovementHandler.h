//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H

#include "glm/mat4x4.hpp"

struct CameraState
{
    glm::mat4& modelMatrix;
    glm::mat4& viewMatrix;
    glm::mat4& projectionMatrix;

    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    float yaw = -90.0f; //make sure camera points towards the negative z-axis by default
    float pitch = 0.0f;

    CameraState(glm::mat4& modelMatrix,
                glm::mat4& viewMatrix,
                glm::mat4& projectionMatrix,
                glm::vec3 cameraPos,
                glm::vec3 cameraFront,
                glm::vec3 cameraUp
    )
        : modelMatrix(modelMatrix),
          viewMatrix(viewMatrix),
          projectionMatrix(projectionMatrix),
          cameraPos(cameraPos),
          cameraFront(cameraFront),
          cameraUp(cameraUp)
    {}
};


class MovementHandler
{
    private:
        const float MOVEMENT_SPEED = 8.0f;
        const float ROTATE_SPEED = 0.8f;

        const glm::vec3 CAMERA_POS = glm::vec3(367.59, 197.453, 328.134);
        const glm::vec3 CAMERA_FRT = glm::vec3(-0.678897, -0.406737, -0.611281);
        const glm::vec3 CAMERA_UP = glm::vec3(0, 1, 0);

        CameraState cameraState;

    public:
        MovementHandler(
            glm::mat4& modelMatrix,
            glm::mat4& viewMatrix,
            glm::mat4& projectionMatrix
        );

        void update();

        void SetCameraPosition(glm::vec3 position);
        void SetCameraFront(glm::vec3 direction);

        void MouseMovement(const glm::ivec2& mouseDelta);

        const CameraState& GetCameraState();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H
