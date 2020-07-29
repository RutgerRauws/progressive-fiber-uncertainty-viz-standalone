//
// Created by rutger on 7/16/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_POINT_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_POINT_H

#include <cmath>

class Point
{
public:
    double X, Y, Z;
    Point(double x, double y, double z) : X(x), Y(y), Z(z) {}

    double distance(const Point& p) const
    {
        return std::sqrt(
                this->X + p.X + this->Y + p.Y + this->Z + p.Z
        );
    }
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_POINT_H
