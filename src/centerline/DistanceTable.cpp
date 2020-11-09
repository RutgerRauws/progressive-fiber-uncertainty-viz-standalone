//
// Created by rutger on 8/17/20.
//

#include "DistanceTable.h"
#include <algorithm>

//Mean of Closest-Point Distances
double DistanceTable::calculateMinimumDistance(const Fiber& fiber1, const Fiber& fiber2)
{
    return
        (calculateMinimumDistance_dm(fiber1, fiber2) + calculateMinimumDistance_dm(fiber2, fiber1))
        /
        2.0f;
}

//Mean of Closest-Point Distances, for internal use only!
double DistanceTable::calculateMinimumDistance_dm(const Fiber& Fi, const Fiber& Fj)
{
    double sum = 0;

    for(const glm::vec3& p_r : Fi.GetUniquePoints())
    {
        auto min_distance = VTK_DOUBLE_MAX;

        for(const glm::vec3& p_s : Fj.GetUniquePoints())
        {
            double distance = glm::distance(p_r, p_s) / 100.0f; //convert distance scores to micrometers
            if(distance < min_distance)
            {
                min_distance = distance;
            }
        }

        sum += min_distance;
    }

    return sum / ((double)Fi.GetUniquePoints().size());
}

bool compareFunc(const DistanceEntry& entry1, const DistanceEntry& entry2)
{
    return entry1.distance < entry2.distance;
}

void DistanceTable::printTable() const
{
    std::cout << "Distance | ID" << std::endl;
    std::cout << "--------------" << std::endl;
    for(const DistanceEntry& entry : entries)
    {
        std::cout << entry.distance << " - " << entry.fiber.get().GetId() << std::endl;
    }
    std::cout << "--------------" << std::endl;
}


DistanceEntry& DistanceTable::InsertNewFiber(const Fiber& newFiber)
{
    double newFiberDistance = 0;

    for(int i = 0; i < entries.size(); i++)
    {
        const Fiber& otherFiber = entries[i].fiber;

        double newDistance = calculateMinimumDistance(newFiber, otherFiber);

//        if(newDistance < DISTANCE_THRESHOLD)
//        {
//            //newFiber and otherFiber are similar enough to consider them identical.
//            //In this case we stop the calculation.
//
//            //TODO: set correctly
//            break;
//        }

        newFiberDistance += newDistance;
        entries[i].distance += newDistance;
    }

    DistanceEntry entry(newFiberDistance, newFiber);
    entries.emplace_back(entry);

    //Todo: We can sort more efficiently by realising that `entries` was already sorted before
    std::sort(entries.begin(), entries.end(), compareFunc);

    return entries.back();
}

const Fiber& DistanceTable::GetCenterline() const
{
    return entries.at(0).fiber;
}