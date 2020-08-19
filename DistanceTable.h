//
// Created by rutger on 8/17/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H


#include <vector>
#include <functional>
#include <unordered_map>
#include "Fiber.h"

struct DistanceEntry
{
    unsigned int* distanceScore_ptr;
    std::reference_wrapper<const Fiber> fiber;

    DistanceEntry(unsigned int* distanceScore_ptr, const Fiber& fiber) : distanceScore_ptr(distanceScore_ptr), fiber(fiber) {};
};

class DistanceTable
{
    private:
        std::vector<DistanceEntry> entries;
        //std::unordered_map<std::reference_wrapper<const Fiber>, double*> lookupTable;

        static double calculateMinimumDistanceScore(const Fiber& fiber1, const Fiber& fiber2);
        static double calculateMinimumDistanceScore_dm(const Fiber& Fi, const Fiber& Fj);

        void printTable() const;

    public:
        void InsertNewFiber(Fiber& fiber);
        const Fiber& GetCenterline() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H
