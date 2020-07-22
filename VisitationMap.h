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

        double voxelSize = 5;

        Voxel** data;

        void initialize();

    public:
        VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);
        explicit VisitationMap(double* bounds);

        ~VisitationMap();

        Voxel* GetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index) const;
        Voxel* GetCell(unsigned int index) const;

        Voxel* FindCell(const Point& point) const;
        Voxel* FindCell(double x, double y, double z) const;

        void SetCell(int x, int y, int z, int value);
        void SetCell(const Point& point, int value);

        unsigned int GetNumberOfCells() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
