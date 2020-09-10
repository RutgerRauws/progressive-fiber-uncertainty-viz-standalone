//
// Created by rutger on 7/2/20.
//

#include "Fiber.h"

unsigned int Fiber::GLOBAL_FIBER_ID = 0;

Fiber::Fiber() : id(++GLOBAL_FIBER_ID) {}

void Fiber::AddPoint(double x, double y, double z)
{
    points.emplace_back(x, y, z);
}

const std::vector<Point>& Fiber::GetPoints() const
{
    return points;
}

unsigned int Fiber::GetId() const
{
    return id;
}

double Fiber::CalculateLength() const
{
    double length = 0.0f;

    for(unsigned int i = 0; i < points.size() - 1; i++)
    {
        length += points[i].distance(points[i + 1]);
    }

    return length;
}
