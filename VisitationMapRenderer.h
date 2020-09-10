//
// Created by rutger on 7/23/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H

#include "VisitationMap.h"
#include "KeyPressObserver.h"
#include <vtkRenderer.h>
#include <vtkPiecewiseFunction.h>
#include <vtkContourValues.h>
#include <vtkImageGaussianSmooth.h>
#include <vtkOpenGLGPUVolumeRayCastMapper.h>

class VisitationMapRenderer : public KeyPressObserver
{
    private:
        static constexpr double SURFACE_TRANSPARENCY = 0.35f;

        VisitationMap& visitationMap;
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkActor> actor;

        vtkSmartPointer<vtkImageGaussianSmooth> smoother;
        vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper> mapper;

        vtkSmartPointer<vtkPiecewiseFunction> opacity;
        vtkSmartPointer<vtkContourValues> isoValues;
        vtkSmartPointer<vtkVolumeProperty> volumeProperty;

        int isovalue;
        bool isSmooth;

        void initialize();
        void updateIsovalue();

        void setSmoothing(bool on);

    public:
        VisitationMapRenderer(VisitationMap& visitationMap,
                              vtkSmartPointer<vtkRenderer> renderer,
                              int startIsovalue = 0,
                              bool isSmooth = false);

        void KeyPressed(const std::basic_string<char>& key) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
