//
// Created by rutger on 7/2/20.
//

#include "Fiber.h"

unsigned int Fiber::GLOBAL_FIBER_ID = 0;

Fiber::Fiber(unsigned int seedPointId)
    : id(++GLOBAL_FIBER_ID),
      seedPointId(seedPointId)
{}

void Fiber::AddSegment(const glm::vec3& p1, const glm::vec3& p2)
{
    if(uniquePoints.empty()) {
        uniquePoints.emplace_back(p1);
    }

    uniquePoints.emplace_back(p2);

    lineSegments.emplace_back(
        LineSegment(seedPointId, id, p1, p2)
    );

    lineSegmentPoints.emplace_back(p1);
    lineSegmentPoints.emplace_back(p2);
}

const std::vector<Fiber::LineSegment>& Fiber::GetLineSegments() const
{
    return lineSegments;
}

const std::vector<glm::vec3> &Fiber::GetLineSegmentsAsPoints() const
{
    return lineSegmentPoints;
}

const std::vector<glm::vec3> &Fiber::GetUniquePoints() const
{
    return uniquePoints;
}

unsigned int Fiber::GetId() const
{
    return id;
}

unsigned int Fiber::GetSeedPointId() const
{
    return seedPointId;
}

double Fiber::CalculateLength() const
{
    double length = 0.0f;

    for(LineSegment lineSegment : lineSegments)
    {
        length += lineSegment.CalculateLength();
    }

    return length;
}