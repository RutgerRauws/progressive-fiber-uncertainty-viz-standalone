//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRACK_BALL_MOVEMENT_HANDLER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRACK_BALL_MOVEMENT_HANDLER_H

#include "MovementHandler.h"

class TrackBallMovementHandler : public MovementHandler
{
    private:
        const float ZOOM_SPEED = 0.08f;
        const float ROTATE_SPEED = 0.008f;

        float vDist, theta, phi;

        glm::vec3 getCartesianCoordinates() const;

    public:
        explicit TrackBallMovementHandler(Camera& camera);

        void MouseMovement(const glm::ivec2& mouseDelta) override;
        void MouseScroll(int delta) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRACK_BALL_MOVEMENT_HANDLER_H
