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
#include <vtkVertexGlyphFilter.h>
#include "FiberObserver.h"
#include "KeyPressObserver.h"

class FiberRenderer : public FiberObserver, public KeyPressObserver
{
    private:
        vtkSmartPointer<vtkRenderer> renderer;

        vtkIdType currentId;
        vtkSmartPointer<vtkPoints> points;
        vtkSmartPointer<vtkCellArray> polyLines;

        vtkSmartPointer<vtkActor> fiberActor;
        vtkSmartPointer<vtkActor> pointsActor;

        vtkSmartPointer<vtkVertexGlyphFilter> vertexGlyphFilter;

        bool fibersShown, pointsShown;

        void initialize();
        
    public:
        explicit FiberRenderer(vtkSmartPointer<vtkRenderer> renderer);
        void NewFiber(Fiber* fiber) override;

        void KeyPressed(const std::basic_string<char>& key) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H
