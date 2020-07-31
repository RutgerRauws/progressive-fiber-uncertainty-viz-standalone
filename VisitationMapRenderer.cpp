//
// Created by rutger on 7/23/20.
//

#include "VisitationMapRenderer.h"

#include <utility>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkOpenGLGPUVolumeRayCastMapper.h>
#include <vtkContourValues.h>

VisitationMapRenderer::VisitationMapRenderer(VisitationMap &visitationMap, vtkSmartPointer<vtkRenderer> renderer)
    : visitationMap(visitationMap),
      renderer(std::move(renderer)),
      actor(vtkSmartPointer<vtkActor>::New()),
      isovalue(0)
{
    initialize();
}

void VisitationMapRenderer::initialize()
{
    //vtkSmartPointer<vtkVolumeRayCastIsosurfaceFunction> b = vtkSmartPointer<vtkVolumeRayCastIsosurfaceFunction>::New();
    //vtkSmartPointer<vtkGPUVolumeRayCastMapper> mapper = vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();
    vtkNew<vtkOpenGLGPUVolumeRayCastMapper> mapper;
    mapper->SetInputData(visitationMap.GetImageData());
    mapper->AutoAdjustSampleDistancesOff();
    mapper->SetSampleDistance(0.5);
    mapper->SetBlendModeToIsoSurface();

    opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();

    vtkSmartPointer<vtkColorTransferFunction> color = vtkSmartPointer<vtkColorTransferFunction>::New();
    color->AddRGBPoint(0, 0, 0, 1);

    vtkSmartPointer<vtkVolumeProperty> property = vtkSmartPointer<vtkVolumeProperty>::New();
    property->SetScalarOpacity(opacity);
    property->SetColor(color);
    property->ShadeOn();
    //property->SetInterpolationTypeToLinear();
    property->SetDiffuse(0.6);
    property->SetSpecular(0.5);
    property->SetAmbient(0.5);

    isoValues = property->GetIsoSurfaceValues();

    vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
    volume->SetMapper(mapper);
    volume->SetProperty(property);

    renderer->AddActor(volume);
}

void VisitationMapRenderer::KeyPressed(const std::basic_string<char>& key)
{
        if(key == "u")
        {
            if(isovalue != UINT_MAX)
            {
                isovalue++;
            }
        }
        else if(key == "j")
        {
            if(isovalue != 0)
            {
                isovalue--;
            }
        }

    std::cout << "isovalue set to " << isovalue << std::endl;
    opacity->RemoveAllPoints();
    opacity->AddPoint(isovalue, SURFACE_TRANSPARENCY);

    isoValues->SetValue(0, isovalue);
}
