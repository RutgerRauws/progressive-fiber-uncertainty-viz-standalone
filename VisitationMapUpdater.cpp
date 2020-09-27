//
// Created by rutger on 7/23/20.
//

#include "VisitationMapUpdater.h"
#include "Cell.h"

VisitationMapUpdater::VisitationMapUpdater(VisitationMap& visitationMap)
        : visitationMap(visitationMap)
{}

void VisitationMapUpdater::NewFiber(Fiber* fiber)
{
    visitationMap.InsertFiber(*fiber);
}

bool VisitationMapUpdater::isCellInsideSphere(const Point& center, double radius, const Point& point, double cellSize)
{
    double halfSize = cellSize / 2.0f;

    if(isPointInsideSphere(center, radius, point.X - halfSize, point.Y - halfSize, point.Z - halfSize)
    || isPointInsideSphere(center, radius, point.X - halfSize, point.Y - halfSize, point.Z + halfSize)
    || isPointInsideSphere(center, radius, point.X - halfSize, point.Y + halfSize, point.Z - halfSize)
    || isPointInsideSphere(center, radius, point.X - halfSize, point.Y + halfSize, point.Z + halfSize)
    || isPointInsideSphere(center, radius, point.X + halfSize, point.Y - halfSize, point.Z - halfSize)
    || isPointInsideSphere(center, radius, point.X + halfSize, point.Y - halfSize, point.Z + halfSize)
    || isPointInsideSphere(center, radius, point.X + halfSize, point.Y + halfSize, point.Z - halfSize)
    || isPointInsideSphere(center, radius, point.X + halfSize, point.Y + halfSize, point.Z + halfSize)
    )
    {
        return true;
    }
    else
    {
        return false;
    }
}


bool VisitationMapUpdater::isPointInsideSphere(const Point& center, double radius, double x, double y, double z)
{
    return pow(x - center.X, 2)
        +  pow(y - center.Y, 2)
        +  pow(z - center.Z, 2) <= std::pow(radius, 2);
}
