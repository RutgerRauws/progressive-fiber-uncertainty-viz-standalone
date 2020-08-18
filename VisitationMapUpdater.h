//
// Created by rutger on 7/23/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H


#include "FiberObserver.h"
#include "Fiber.h"
#include "VisitationMap.h"

class VisitationMapUpdater : public FiberObserver {
    private:
        VisitationMap& visitationMap;

    public:
        explicit VisitationMapUpdater(VisitationMap& visitationMap);
        void NewFiber(Fiber* fiber) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
