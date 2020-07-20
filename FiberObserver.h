//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_OBSERVER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_OBSERVER_H

#include "Fiber.h"

class FiberObserver
{
    public:
        virtual void NewFiber(const Fiber& fiber) = 0;
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_OBSERVER_H
