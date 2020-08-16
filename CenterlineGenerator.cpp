//
// Created by rutger on 7/29/20.
//

#include <functional>
#include "CenterlineGenerator.h"

CenterlineGenerator::CenterlineGenerator(CenterlineRenderer& centerlineRenderer)
    : centerlineRenderer(centerlineRenderer), centerfiber_ptr(nullptr)
{}

bool compareFunc(const std::pair<double, Fiber*>& pair1, const std::pair<double, Fiber*>& pair2)
{
    return pair1.first < pair2.first;
}

void CenterlineGenerator::NewFiber(Fiber* newFiber)
{
    double newFiberDistance = 0;

    for(int i = 0; i < distanceTable.size(); i++)
    {
        const Fiber* const otherFiber = distanceTable[i].second;

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
        distanceTable[i].first += newDistance;
    }

    distanceTable.emplace_back(newFiberDistance, newFiber);
    std::sort(distanceTable.begin(), distanceTable.end(), compareFunc);

    if(centerfiber_ptr == nullptr || centerfiber_ptr != GetCenterline())
    {
        centerfiber_ptr = GetCenterline();
        centerlineRenderer.get().Update(GetCenterline());
    }
}

//Mean of Closest-Point Distances
double CenterlineGenerator::calculateMinimumDistanceScore(const Fiber* const fiber1, const Fiber* const fiber2)
{
    return
        (calculateMinimumDistanceScore_dm(fiber1, fiber2) + calculateMinimumDistanceScore_dm(fiber2, fiber1))
        /
        2.0f;
}

//Mean of Closest-Point Distances, for internal use only!
double CenterlineGenerator::calculateMinimumDistanceScore_dm(const Fiber* const Fi, const Fiber* const Fj)
{
    double sum = 0;

    for(const Point& p_r : Fi->GetPoints())
    {
        double min_distance = VTK_DOUBLE_MAX;
        for(const Point& p_s : Fj->GetPoints())
        {
            double distance = p_r.distance(p_s);
            if(distance < min_distance)
            {
                min_distance = distance;
            }
        }

        sum += min_distance;
    }

    return sum / ((double)Fi->GetPoints().size());
}

const Fiber* CenterlineGenerator::GetCenterline() const
{
    return distanceTable.at(0).second;
}