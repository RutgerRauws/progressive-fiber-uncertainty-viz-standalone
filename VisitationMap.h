//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkImageData.h>
#include <vtkGaussianSplatter.h>
#include "Point.h"
#include "Cell.h"

class VisitationMap
{
    private:
        double cellSize;

        vtkSmartPointer<vtkPolyData> vtkData;
        vtkSmartPointer<vtkGaussianSplatter> splatter;

        //vtkSmartPointer<vtkUnsignedIntArray> fiberIds;
        vtkSmartPointer<vtkUnsignedIntArray> frequencies;
        vtkSmartPointer<vtkDoubleArray> distanceScores;

        unsigned int* start_ptr = nullptr;
        Cell** data;

        void initialize();
        void cellModifiedCallback();

        static bool isInCell(const double* cellCenterPoint, const Point& point, double cellSize);

    public:
        explicit VisitationMap(double cellSize);

//        void GetIndex(const Point& point, unsigned int* x_index, unsigned int* y_index, unsigned int* z_index) const;
//
//        Cell* GetCell(unsigned int index) const;
//        Cell* GetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index) const;
//        Cell* GetCell(const Point& point) const;

        void InsertPoint(const Point& point) const;
        unsigned int GetFrequency(const Point& point) const;

        unsigned int GetNumberOfCells() const;
        double GetCellSize() const;

        vtkSmartPointer<vtkPolyData> GetVTKData() const;
        vtkAlgorithmOutput* GetImageOutput() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
