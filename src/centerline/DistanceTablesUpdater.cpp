//
// Created by rutger on 11/6/20.
//

#include "DistanceTablesUpdater.h"

DistanceTablesUpdater::DistanceTablesUpdater(unsigned int numberOfSeedPoints)
    : numberOfSeedPoints(numberOfSeedPoints)
{
    initialize();
}

void DistanceTablesUpdater::initialize()
{
    distanceTables.reserve(numberOfSeedPoints);

    for(unsigned int i = 0; i < numberOfSeedPoints; i++)
    {
        distanceTables.emplace_back(DistanceTable(i));
    }
}

void DistanceTablesUpdater::NewFiber(Fiber* fiber)
{
    DistanceTable& distanceTable = distanceTables.at(fiber->GetSeedPointId());

    DistanceEntry& newDistanceEntry = distanceTable.InsertNewFiber(*fiber);
    distanceScores.push_back(&newDistanceEntry.distance);
}