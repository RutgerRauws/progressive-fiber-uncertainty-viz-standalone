//
// Created by rutger on 8/17/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H


#include <vector>
#include <functional>
#include "../util/Fiber.h"

struct DistanceEntry
{
    double distance;
    std::reference_wrapper<const Fiber> fiber;

    DistanceEntry(double distance, const Fiber& fiber) : distance(distance), fiber(fiber) {};
};

struct DistanceEntryGL
{
    GLdouble distance;           //16 bytes
    GLuint   fiberId;            // 4 bytes
    GLuint   seedPointId;        // 4 bytes
    GLuint   padding1, padding2; // 8 bytes
};

class DistanceTable
{
    private:
        unsigned int seedPointId;
        std::vector<DistanceEntry> entries;

        static double calculateMinimumDistance(const Fiber& fiber1, const Fiber& fiber2);
        static double calculateMinimumDistance_dm(const Fiber& Fi, const Fiber& Fj);

        void printTable() const;

    public:
        explicit DistanceTable(unsigned int seedPointId) : seedPointId(seedPointId) {}

        DistanceEntry& InsertNewFiber(const Fiber& fiber);
        const Fiber& GetCenterline() const;

        std::vector<DistanceEntryGL> ToGLFormat() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DISTANCE_TABLE_H
