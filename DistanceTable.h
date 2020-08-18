//
// Created by rutger on 8/17/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H


#include <vector>
#include <functional>
#include "Fiber.h"

struct DistanceEntry
{
    double distance;
    std::reference_wrapper<const Fiber> fiber;

    DistanceEntry(double distance, const Fiber& fiber) : distance(distance), fiber(fiber) {};
};

class DistanceTable
{
    private:
        std::vector<DistanceEntry> entries;

        static double calculateMinimumDistanceScore(const Fiber& fiber1, const Fiber& fiber2);
        static double calculateMinimumDistanceScore_dm(const Fiber& Fi, const Fiber& Fj);

        void printTable() const;

    public:
        void InsertNewFiber(const Fiber& fiber);
        const Fiber& GetCenterline() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H
