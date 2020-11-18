//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H

#include <src/util/Camera.h>
#include "glm/mat4x4.hpp"

class MovementHandler
{
    private:
        const float MOVEMENT_SPEED = 8.0f;
        const float ROTATE_SPEED = 0.8f;

        Camera& camera;

    public:
        MovementHandler(Camera& camera);

        void MouseMovement(const glm::ivec2& mouseDelta);
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H
