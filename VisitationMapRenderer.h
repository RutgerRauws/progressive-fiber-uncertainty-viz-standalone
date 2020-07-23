//
// Created by rutger on 7/23/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H


#include "VisitationMap.h"
#include <vtkRenderer.h>

class VisitationMapRenderer
{
    private:
        VisitationMap& visitationMap;
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkActor> actor;

        void initialize();

    public:
        VisitationMapRenderer(VisitationMap& visitationMap, vtkSmartPointer<vtkRenderer> renderer);
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
