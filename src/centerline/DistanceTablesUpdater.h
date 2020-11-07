//
// Created by rutger on 11/6/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLES_UPDATER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLES_UPDATER_H


#include "../util/FiberObserver.h"
#include "DistanceTable.h"

using DistanceTableCollection = std::vector<DistanceTable>;

class DistanceTablesUpdater : public FiberObserver
{
private:
    unsigned int numberOfSeedPoints;

    DistanceTableCollection distanceTables;
    std::vector<double*> distanceScores;

    void initialize();

public:
    explicit DistanceTablesUpdater(unsigned int numberOfSeedPoints);

    void NewFiber(Fiber* fiber) override;

    const DistanceTableCollection& GetDistanceTables() const { return distanceTables; };

    std::vector<double*> GetDistanceScores() const { return distanceScores; };
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLES_UPDATER_H
