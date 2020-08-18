//
// Created by rutger on 7/20/20.
//

#include "Cell.h"
#include "VisitationMap.h"
#include <vtkCellData.h>

Cell::Cell(Point position,
           double size,
           unsigned int* value_ptr,
           VisitationMap* visitationMap,
           void (VisitationMap::*modifiedCallback)()
           )
        : value_ptr(value_ptr),
          position(position),
          size(size),
          visitationMap(visitationMap),
          modifiedCallback(modifiedCallback)
{
    if(value_ptr == nullptr)
    {
        std::cerr << "Assigned value pointer is nullptr" << std::endl;
    }
}

void Cell::updateValue()
{
    *value_ptr = fibers.size();
}

unsigned int Cell::GetValue() const
{
    if(value_ptr == nullptr)
    {
        std::cerr << "Tried to get a value of a uninitialized visitation-map cell!" << std::endl;
        return 0;
    }

    return *value_ptr;
}

//void Cell::SetValue(unsigned int val)
//{
//    if(value_ptr == nullptr)
//    {
//        throw std::invalid_argument("Cannot set value to incorrectly initialized Cell!");
//    }
//
//    *value_ptr = val;
//    (visitationMap->*modifiedCallback)(); //Telling the visitation map that the vtkImageData object was modified
//}

void Cell::InsertFiber(const Fiber &fiber)
{
    fibers.emplace_back(fiber);
    updateValue();
}


Point Cell::GetPosition() const
{
    return position;
}

void Cell::GetBounds(double* bounds) const
{
    double halfSize = size / 2.0f;

    bounds[0] = position.X - halfSize; //xMin
    bounds[1] = position.X + halfSize; //xMax
    bounds[2] = position.Y - halfSize; //yMin
    bounds[3] = position.Y + halfSize; //yMax
    bounds[4] = position.Z - halfSize; //zMin
    bounds[5] = position.Z + halfSize; //zMax
}

bool Cell::Contains(const Point& point) const
{
    double halfSize = size / 2.0f;
    double xmin = position.X - halfSize;
    double xmax = position.X + halfSize;
    double ymin = position.Y - halfSize;
    double ymax = position.Y + halfSize;
    double zmin = position.Z - halfSize;
    double zmax = position.Z + halfSize;

    return (xmin <= point.X) && (point.X <= xmax)
        && (ymin <= point.Y) && (point.Y <= ymax)
        && (zmin <= point.Z) && (point.Z <= zmax);
}

bool Cell::Contains(const Fiber& otherFiber) const
{
    for(const Fiber& fiber : fibers)
    {
        if(fiber.GetId() == otherFiber.GetId())
        {
            return true;
        }
    }

    return false;
}
