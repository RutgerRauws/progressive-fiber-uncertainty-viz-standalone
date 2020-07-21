//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H


#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include "VisitationMap.h"
#include "FiberObserver.h"

class VisitationMapRenderer : public FiberObserver
{
    private:
        VisitationMap& visitationMap;

        vtkSmartPointer<vtkRenderWindow> renderWindow;

public:
        VisitationMapRenderer(VisitationMap& visitationMap,
                              vtkSmartPointer<vtkRenderer> renderer,
                              vtkSmartPointer<vtkRenderWindow>);

        void NewFiber(const Fiber& fiber) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
