//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H

#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include "Point.h"
#include "Voxel.h"

class VisitationMap {
    private:
        double xmin,xmax, ymin,ymax, zmin,zmax;

        int width;
        int height;
        int depth;

        double voxelSize = 10;

        Voxel** data;

        void initialize();

    public:
        VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);
        explicit VisitationMap(double* bounds);

        ~VisitationMap();

        Voxel* GetCell(int x, int y, int z) const;
        Voxel* GetCell(const Point& point) const;
        Voxel* GetCell(unsigned int index) const;

        void SetCell(int x, int y, int z, int value);
        void SetCell(const Point& point, int value);

        unsigned int GetNumberOfCells() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
