//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VOXEL_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VOXEL_H

#include <vtkCubeSource.h>
#include "Point.h"

class Voxel {
    private:
    int value;
        Point position;
        double size;

        vtkSmartPointer<vtkCubeSource> cubeSource;

        void updateVTKObject();

    public:
        Voxel();
        Voxel(Point position, double size, int value);

        int GetValue() const;
        void SetValue(int value);

        Point GetPosition() const;
        void GetBounds(double* bounds) const;

        vtkSmartPointer<vtkCubeSource> GetVTKObject() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VOXEL_H
