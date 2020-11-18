//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H

#include <src/util/Camera.h>

class MovementHandler
{
    protected:
        Camera& camera;

    public:
        explicit MovementHandler(Camera& camera)
            : camera(camera)
        {};

        virtual void MouseMovement(const glm::ivec2& mouseDelta) = 0;
        virtual void MouseScroll(int delta) = 0;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_MOVEMENT_HANDLER_H
