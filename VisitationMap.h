//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H

#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkImageData.h>
#include "Point.h"
#include "Voxel.h"

class VisitationMap {
    private:
        double xmin,xmax, ymin,ymax, zmin,zmax;

        int width;
        int height;
        int depth;

        double voxelSize = 5;

        vtkSmartPointer<vtkImageData> imageData;
        unsigned int* start_ptr = nullptr;

        void initialize();

        unsigned int* getCell(unsigned int x_index, unsigned int y_index, unsigned int z_index) const;
        unsigned int* getCell(unsigned int index) const;

        unsigned int* findCell(const Point& point) const;

        bool containedInCell(unsigned int x_index, unsigned int y_index, unsigned int z_index, const Point& point) const;

    public:
        VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);
        explicit VisitationMap(double* bounds);

        unsigned int GetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index) const;
        unsigned int GetCell(unsigned int index) const;

        unsigned int FindCell(const Point& point) const;
        unsigned int FindCell(double x, double y, double z) const;

        void SetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index, unsigned int value);
        void SetCell(const Point& point, unsigned int value);

        unsigned int GetNumberOfCells() const;

        vtkSmartPointer<vtkImageData> GetImageData() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
