//
// Created by rutger on 11/6/20.
//

#include "DistanceTablesUpdater.h"

DistanceTablesUpdater::DistanceTablesUpdater(GL& gl, unsigned int numberOfSeedPoints)
    : distanceTables(gl, numberOfSeedPoints)
{}

void DistanceTablesUpdater::NewFiber(Fiber* fiber)
{
    distanceTables.InsertFiber(fiber->GetSeedPointId(), fiber);
}