//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VOXEL_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VOXEL_H

#include <vtkCubeSource.h>
#include "Point.h"

class VisitationMap;

class Voxel {
    private:
        unsigned int id;

        unsigned int* value_ptr;
        Point position;
        double size;

        vtkSmartPointer<vtkCubeSource> cubeSource;

        VisitationMap* visitationMap;

        void updateVTKObject();

    public:
        Voxel() = delete;
        Voxel(VisitationMap* visitationMap, Point position, double size, unsigned int* value_ptr);

        unsigned int GetValue() const;
        void SetValue(unsigned int value);

        Point GetPosition() const;
        void GetBounds(double* bounds) const;

        bool Contains(const Point& point) const;

        vtkSmartPointer<vtkCubeSource> GetVTKObject() const;
        unsigned int GetID() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VOXEL_H
