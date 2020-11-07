//
// Created by rutger on 11/6/20.
//

#include "DistanceTablesUpdater.h"

DistanceTablesUpdater::DistanceTablesUpdater(unsigned int numberOfSeedPoints)
    : distanceTables(numberOfSeedPoints)
{}

void DistanceTablesUpdater::NewFiber(Fiber* fiber)
{
    distanceTables.InsertFiber(fiber->GetSeedPointId(), fiber);
}