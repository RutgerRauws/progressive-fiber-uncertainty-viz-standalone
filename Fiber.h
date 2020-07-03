//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>

class Fiber
{
    private:
        vtkSmartPointer<vtkPoints> points;
        vtkSmartPointer<vtkIdList> ids;
        vtkSmartPointer<vtkCellArray> vertices;
        
    public:
        Fiber();
        
        void AddPoint(double x, double y, double z);
        vtkSmartPointer<vtkPoints> GetPoints() const;
        vtkSmartPointer<vtkIdList> GetIds() const;
    
        vtkSmartPointer<vtkPolyLine> CreatePolyLine() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
