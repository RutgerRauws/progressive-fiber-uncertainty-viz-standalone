//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H

#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkPolyDataMapper.h>
#include "FiberObserver.h"

class FiberRenderer : public FiberObserver
{
    private:
        vtkIdType currentId;
        vtkSmartPointer<vtkPoints> points;
        vtkSmartPointer<vtkCellArray> polyLines;
        vtkSmartPointer<vtkPolyData> polyData;
        vtkSmartPointer<vtkPolyDataMapper> mapper;
        vtkSmartPointer<vtkActor> actor;
    
        vtkSmartPointer<vtkRenderer> renderer;
    
        void initialize();
        
    public:
        explicit FiberRenderer(vtkSmartPointer<vtkRenderer> renderer);
        void NewFiber(const Fiber& fiber) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H
