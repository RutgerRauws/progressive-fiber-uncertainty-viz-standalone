//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FPS_MOVEMENT_HANDLER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FPS_MOVEMENT_HANDLER_H

#include "MovementHandler.h"

class FPSMovementHandler : public MovementHandler
{
    private:
        const float MOVEMENT_SPEED = 8.0f;
        const float ROTATE_SPEED = 0.8f;

        const glm::vec3 CAMERA_POS = glm::vec3(367.59, 197.453, 328.134);
        const glm::vec3 CAMERA_FRT = glm::vec3(-0.678897, -0.406737, -0.611281);
        const glm::vec3 CAMERA_UP  = glm::vec3(0, 1, 0);

    public:
        FPSMovementHandler(Camera& camera);

        void MouseMovement(const glm::ivec2& mouseDelta) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FPS_MOVEMENT_HANDLER_H
