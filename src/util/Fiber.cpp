//
// Created by rutger on 7/2/20.
//

#include "Fiber.h"

unsigned int Fiber::GLOBAL_FIBER_ID = 0;

Fiber::Fiber(unsigned int seedPointId)
    : id(++GLOBAL_FIBER_ID),
      seedPointId(seedPointId)
{}

void Fiber::AddPoint(double x, double y, double z)
{
    points.emplace_back(glm::vec4(x, y, z, 1));
}

const std::vector<glm::vec4>& Fiber::GetPoints() const
{
    return points;
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

    for(unsigned int i = 0; i < points.size() - 1; i++)
    {
        length += glm::distance(points[i], points[i + 1]);
    }

    return length;
}