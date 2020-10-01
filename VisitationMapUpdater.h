//
// Created by rutger on 7/23/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H


#include "FiberObserver.h"
#include "Fiber.h"
#include "VisitationMap.h"

class VisitationMapUpdater : public FiberObserver
{
    private:
        VisitationMap& visitationMap;
        VisitationMap& visitationMapSplatted;

        double splatKernelRadius;

        void splat(const Point& p, double radius, Fiber* fiber);
        static bool isCellInsideSphere(const Point& center, double radius, const Point& point, double cellSize);
        static bool isPointInsideSphere(const Point& center, double radius, double x, double y, double z);

public:
        VisitationMapUpdater(VisitationMap& visitationMap, VisitationMap& visitationMapSplatted, double splatKernelRadius);
        void NewFiber(Fiber* fiber) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
