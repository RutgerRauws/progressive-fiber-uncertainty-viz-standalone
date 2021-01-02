//
// Created by rutger on 11/7/20.
//

#include "DistanceTableCollection.h"

DistanceTableCollection::DistanceTableCollection(GL& gl, unsigned int numberOfSeedPoints)
    : smallestDistanceScore(std::numeric_limits<double>::max()),
      largestDistanceScore(std::numeric_limits<double>::min())
{
    distanceTables.reserve(numberOfSeedPoints);

    for(unsigned int i = 0; i < numberOfSeedPoints; i++)
    {
        distanceTables.emplace_back(DistanceTable(i));
    }

    gl.glGenBuffers(1, &distance_scores_ssbo_id);
}

bool comparisonDistanceScorePtrs(double* score_ptr, double* score2_ptr)
{
    return *score_ptr < *score2_ptr;
}

void DistanceTableCollection::InsertFiber(unsigned int seedPointId, Fiber* fiber)
{
    DistanceTable& distanceTable = distanceTables.at(seedPointId);

    mtx.lock();
    DistanceEntry* newDistanceEntry = distanceTable.InsertNewFiber(*fiber);

    distanceScores.resize( fiber->GetId() + 1, nullptr);
    distanceScores.at(fiber->GetId()) = &(newDistanceEntry->distance);

    sortedDistanceScores.push_back(&(newDistanceEntry->distance));
    std::sort(sortedDistanceScores.begin(), sortedDistanceScores.end(), comparisonDistanceScorePtrs);

    if(newDistanceEntry->distance < smallestDistanceScore)
    {
        smallestDistanceScore = newDistanceEntry->distance;
    }
    else if(newDistanceEntry->distance > largestDistanceScore)
    {
        largestDistanceScore = newDistanceEntry->distance;
    }


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

double DistanceTableCollection::GetDistanceScoreForPercentage(float percentage) const
{
    unsigned int numberOfFibers = sortedDistanceScores.size();

    if(numberOfFibers == 0)
    {
        return 0;
    }

    int index = percentage * numberOfFibers - 1;

    if(index < 0)
    {
        return 0;
    }

    return *sortedDistanceScores[index];
}
