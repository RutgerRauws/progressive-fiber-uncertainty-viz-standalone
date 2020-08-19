//
// Created by rutger on 7/20/20.
//

#include "Cell.h"
#include "VisitationMap.h"
#include <vtkCellData.h>

Cell::Cell(Point position,
           double size,
           unsigned int* fiberFrequency_ptr,
           double* fiberDistanceScore_ptr,
           VisitationMap* visitationMap,
           void (VisitationMap::*modifiedCallback)()
           )
        : fiberFrequency_ptr(fiberFrequency_ptr),
          fiberDistanceScore_ptr(fiberDistanceScore_ptr),
          position(position),
          size(size),
          visitationMap(visitationMap),
          modifiedCallback(modifiedCallback)
{
    if(fiberFrequency_ptr == nullptr)
    {
        std::cerr << "Assigned fiber frequency pointer is nullptr" << std::endl;
    }

    if(fiberDistanceScore_ptr == nullptr)
    {
        std::cerr << "Assigned fiber distance score pointer is nullptr" << std::endl;
    }
}

void Cell::updateFiberFrequency()
{
    *fiberFrequency_ptr = fibers.size();
    (visitationMap->*modifiedCallback)(); //Telling the visitation map that the vtkImageData object was modified
}

void Cell::updateMinimumDistanceScore()
{
    //We have to iterate over all fibers in the cell, as fibers other than the latest added fiber may have a lower
    //distance score.

    for(const Fiber& fiber : fibers)
    {
        if(fiber.GetDistanceScore() < GetMinimumDistanceScore())
        {
            *fiberDistanceScore_ptr = fiber.GetDistanceScore();

            if(*fiberDistanceScore_ptr != 0)
            {
                std::cout << *fiberDistanceScore_ptr << std::endl;
            }
        }
    }

    (visitationMap->*modifiedCallback)(); //Telling the visitation map that the vtkImageData object was modified
}

unsigned int Cell::GetFiberFrequency() const
{
    if(fiberFrequency_ptr == nullptr)
    {
        std::cerr << "Tried to get a fiber frequency of a uninitialized visitation-map cell!" << std::endl;
        return 0;
    }

    return *fiberFrequency_ptr;
}

double Cell::GetMinimumDistanceScore() const
{
    if(fiberDistanceScore_ptr == nullptr)
    {
        std::cerr << "Tried to get a distance score of a uninitialized visitation-map cell!" << std::endl;
        return 0;
    }

    return *fiberDistanceScore_ptr;
}

//void Cell::SetValue(unsigned int val)
//{
//    if(fiberFrequency_ptr == nullptr)
//    {
//        throw std::invalid_argument("Cannot set value to incorrectly initialized Cell!");
//    }
//
//    *fiberFrequency_ptr = val;
//    (visitationMap->*modifiedCallback)(); //Telling the visitation map that the vtkImageData object was modified
//}

void Cell::InsertFiber(const Fiber& fiber)
{
    fibers.emplace_back(fiber);
    updateFiberFrequency();
    updateMinimumDistanceScore();
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
