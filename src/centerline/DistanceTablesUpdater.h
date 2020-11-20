//
// Created by rutger on 11/6/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLES_UPDATER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLES_UPDATER_H

#include "../util/FiberObserver.h"
#include "DistanceTableCollection.h"

class DistanceTablesUpdater : public FiberObserver
{
private:
    DistanceTableCollection distanceTables;

public:
    DistanceTablesUpdater(GL& gl, unsigned int numberOfSeedPoints);

    void NewFiber(Fiber* fiber) override;

    const DistanceTableCollection& GetDistanceTables() const { return distanceTables; };

};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLES_UPDATER_H
