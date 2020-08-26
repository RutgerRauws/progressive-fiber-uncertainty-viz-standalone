//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H

#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkImageData.h>
#include "Point.h"
#include "Cell.h"

class VisitationMap
{
    private:
        double xmin,xmax, ymin,ymax, zmin,zmax;

        int width;
        int height;
        int depth;

        double cellSize = 3;

        vtkSmartPointer<vtkImageData> imageData;
        unsigned int* start_ptr = nullptr;

        Cell** data;

        void initialize();
        void cellModifiedCallback();

    public:
        VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);
        explicit VisitationMap(double* bounds);

        Cell* GetCell(unsigned int index) const;
        Cell* FindCell(const Point& point) const;

        unsigned int GetNumberOfCells() const;

        vtkSmartPointer<vtkImageData> GetImageData() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
