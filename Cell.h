//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_H

#include <vtkCubeSource.h>
#include <functional>
#include "Point.h"
#include "Fiber.h"

class VisitationMap;

class Cell
{
    private:
        unsigned int* value_ptr;
        Point position;
        double size;

        std::vector<std::reference_wrapper<const Fiber>> fibers;

        VisitationMap* visitationMap;
        void (VisitationMap::*modifiedCallback)();

        void updateValue();

public:
        Cell() = delete;
        Cell(Point position,
             double size,
             unsigned int* value_ptr,
             VisitationMap* visitationMap,
             void (VisitationMap::*modifiedCallback)()
        );

        unsigned int GetValue() const;
        //void SetValue(unsigned int value);
        void InsertFiber(const Fiber& fiber);

        Point GetPosition() const;
        void GetBounds(double* bounds) const;

        bool Contains(const Point& point) const;
        bool Contains(const Fiber& fiber) const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_H
