//
// Created by rutger on 11/7/20.
//

#include <GL/glew.h>
#include "DistanceTableCollection.h"

DistanceTableCollection::DistanceTableCollection(unsigned int numberOfSeedPoints)
{
    distanceTables.reserve(numberOfSeedPoints);

    for(unsigned int i = 0; i < numberOfSeedPoints; i++)
    {
        distanceTables.emplace_back(DistanceTable(i));
    }

    glGenBuffers(1, &distance_scores_ssbo_id);
}

void DistanceTableCollection::InsertFiber(unsigned int seedPointId, Fiber* fiber)
{
    DistanceTable& distanceTable = distanceTables.at(seedPointId);

    mtx.lock();
    DistanceEntry* newDistanceEntry = distanceTable.InsertNewFiber(*fiber);

    distanceScores.resize(fiber->GetId() + 1, nullptr);
    distanceScores.at(fiber->GetId()) = &(newDistanceEntry->distance);
    mtx.unlock();
}

const DistanceTable& DistanceTableCollection::GetDistanceTable(unsigned int seedPointId) const
{
    return distanceTables.at(seedPointId);
}

unsigned int DistanceTableCollection::GetNumberOfSeedPoints() const
{
    return distanceTables.size();
}

std::vector<double> DistanceTableCollection::GetDistanceScoreCopy() const
{
    unsigned int numberOfScores = distanceScores.size();

    std::vector<double> distanceScoresCopy;
    distanceScoresCopy.reserve(numberOfScores);

    for(int i = 0; i < numberOfScores; i++)
    {
        double* distanceScorePtr = distanceScores[i];

        if(distanceScorePtr == nullptr)
        {
            distanceScoresCopy.push_back(std::numeric_limits<double>::max());
        }
        else
        {
            distanceScoresCopy.push_back(*distanceScorePtr);
        }
    }

    return distanceScoresCopy;
}