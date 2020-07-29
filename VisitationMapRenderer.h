//
// Created by rutger on 7/23/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H


#include "VisitationMap.h"
#include "IsovalueObserver.h"
#include <vtkRenderer.h>
#include <vtkPiecewiseFunction.h>
#include <vtkContourValues.h>

class VisitationMapRenderer : public IsovalueObserver
{
    private:
        static constexpr double SURFACE_TRANSPARENCY = 0.6f;

        VisitationMap& visitationMap;
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkActor> actor;

        vtkSmartPointer<vtkPiecewiseFunction> opacity;
        vtkSmartPointer<vtkContourValues> isoValues;

        void initialize();

    public:
        VisitationMapRenderer(VisitationMap& visitationMap, vtkSmartPointer<vtkRenderer> renderer);
        void NewIsovalue(unsigned int value) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
