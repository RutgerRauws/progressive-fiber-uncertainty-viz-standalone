//
// Created by rutger on 11/7/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_COLLECTION_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_COLLECTION_H


#include <vector>
#include "DistanceTable.h"

class DistanceTableCollection
{
private:
    std::vector<DistanceTable> distanceTables;
    std::vector<double*> distanceScores;

public:
    explicit DistanceTableCollection(unsigned int numberOfSeedPoints);

    void InsertFiber(unsigned int seedPointId, Fiber* fiber);

    const DistanceTable& GetDistanceTable(unsigned int seedPointId) const;
    unsigned int GetNumberOfSeedPoints() const;

    std::vector<double> GetDistanceScoreCopy() const;
    unsigned int GetNumberOfBytes() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_COLLECTION_H
