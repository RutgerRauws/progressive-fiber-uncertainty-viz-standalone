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
#include "Point.h"
#include "Cell.h"
#include "GaussianFiberSplatter.h"
#include "FiberObserver.h"

class VisitationMap
{
    private:
        double cellSize;

        vtkSmartPointer<vtkPolyData> vtkData;
        vtkSmartPointer<GaussianFiberSplatter> splatter;

        vtkSmartPointer<vtkUnsignedIntArray> frequencies;
        vtkSmartPointer<vtkDoubleArray> distanceScores;
        std::vector<std::vector<std::reference_wrapper<const Fiber>>> fibers;

        void initialize();
        void insertPoint(const Point& point, const Fiber& fiber);

        void updateBounds();

        static bool isInCell(const double* cellCenterPoint, const Point& point, double cellSize);

    public:
        explicit VisitationMap(double cellSize);

        void InsertFiber(const Fiber& fiber);
        unsigned int GetFrequency(const Point& point) const;

        unsigned int GetNumberOfCells() const;
        double GetCellSize() const;

        vtkSmartPointer<vtkPolyData> GetVTKData() const;
        vtkAlgorithmOutput* GetImageOutput() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
