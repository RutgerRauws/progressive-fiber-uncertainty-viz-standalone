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
    glm::vec4 p1_w = glm::vec4(p1, 1);
    glm::vec4 p2_w = glm::vec4(p2, 1);

    if(uniquePoints.empty()) {
        uniquePoints.emplace_back(p1_w);
    }

    uniquePoints.emplace_back(p2_w);

    lineSegments.emplace_back(
        LineSegment(p1_w, p2_w)
    );

    lineSegmentPoints.emplace_back(p1_w);
    lineSegmentPoints.emplace_back(p2_w);
}

const std::vector<Fiber::LineSegment>& Fiber::GetLineSegments() const
{
    return lineSegments;
}

const std::vector<glm::vec4> &Fiber::GetLineSegmentsAsPoints() const
{
    return lineSegmentPoints;
}

const std::vector<glm::vec4> &Fiber::GetUniquePoints() const
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