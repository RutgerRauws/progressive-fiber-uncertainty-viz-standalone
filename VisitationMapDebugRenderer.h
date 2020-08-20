//
// Created by rutger on 7/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_DEBUG_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_DEBUG_RENDERER_H

#ifdef VISITATION_MAP_CELL_DEBUG

#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include "VisitationMap.h"
#include "FiberObserver.h"

class VisitationMapDebugRenderer
{
    private:
        VisitationMap& visitationMap;
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkActor> actor;

        void initialize();

public:
        VisitationMapDebugRenderer(VisitationMap& visitationMap,
                                   vtkSmartPointer<vtkRenderer> renderer);
};

#endif //VISITATION_MAP_CELL_DEBUG

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_DEBUG_RENDERER_H
