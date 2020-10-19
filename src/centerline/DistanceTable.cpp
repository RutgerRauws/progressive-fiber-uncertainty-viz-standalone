//
// Created by rutger on 8/17/20.
//

#include "DistanceTable.h"
#include <functional>
#include <algorithm>

//Mean of Closest-Point Distances
double DistanceTable::calculateMinimumDistanceScore(const Fiber& fiber1, const Fiber& fiber2)
{
    return
        (calculateMinimumDistanceScore_dm(fiber1, fiber2) + calculateMinimumDistanceScore_dm(fiber2, fiber1))
        /
        2.0f;
}

//Mean of Closest-Point Distances, for internal use only!
double DistanceTable::calculateMinimumDistanceScore_dm(const Fiber& Fi, const Fiber& Fj)
{
    double sum = 0;

    for(const glm::vec4& p_r : Fi.GetPoints())
    {
        auto min_distance = VTK_DOUBLE_MAX;
        for(const glm::vec4& p_s : Fj.GetPoints())
        {
            double distance = glm::distance(p_r, p_s);
            if(distance < min_distance)
            {
                min_distance = distance;
            }
        }

        sum += min_distance;
    }

    return sum / ((double)Fi.GetPoints().size());
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


void DistanceTable::InsertNewFiber(const Fiber& newFiber)
{
    double newFiberDistance = 0;

    for(int i = 0; i < entries.size(); i++)
    {
        const Fiber& otherFiber = entries[i].fiber;

        double newDistance = calculateMinimumDistanceScore(newFiber, otherFiber);

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

    std::sort(entries.begin(), entries.end(), compareFunc);
}

const Fiber& DistanceTable::GetCenterline() const
{
    return entries.at(0).fiber;
}