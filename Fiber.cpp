//
// Created by rutger on 7/2/20.
//

#include "Fiber.h"

Fiber::Fiber() : points() {}

void Fiber::AddPoint(double x, double y, double z)
{
    points.emplace_back(x, y, z);
}

const std::vector<Point>& Fiber::GetPoints() const
{
    return points;
}