//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H

#include <list>
#include "vtkSmartPointer.h"
#include "Point.h"

class Fiber
{
    private:
        std::list<Point> points;
        
    public:
        Fiber();

        void AddPoint(double x, double y, double z);
        const std::list<Point>& GetPoints() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
