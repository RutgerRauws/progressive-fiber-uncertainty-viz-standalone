//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_H

#include <vtkCubeSource.h>
#include "Point.h"

class VisitationMap;

class Cell
{
    private:
        unsigned int* value_ptr;
        Point position;
        double size;

        VisitationMap* visitationMap;
        void (VisitationMap::*modifiedCallback)();

public:
        Cell() = delete;
        Cell(Point position,
             double size,
             unsigned int* value_ptr,
             VisitationMap* visitationMap,
             void (VisitationMap::*modifiedCallback)()
        );

        unsigned int GetValue() const;
        void SetValue(unsigned int value);

        Point GetPosition() const;
        void GetBounds(double* bounds) const;

        bool Contains(const Point& point) const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_H
