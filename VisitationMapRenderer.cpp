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
      isovalue(0),
      isSmooth(false)
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
    mapper->SetSampleDistance(0.01f);
    mapper->SetBlendModeToIsoSurface();

    opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();

    vtkSmartPointer<vtkColorTransferFunction> color = vtkSmartPointer<vtkColorTransferFunction>::New();
    color->AddRGBPoint(0, 0, 0, 1);

    volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
    volumeProperty->SetScalarOpacity(opacity);
    volumeProperty->SetColor(color);
    volumeProperty->ShadeOn();
    volumeProperty->SetDiffuse(0.6);
    volumeProperty->SetSpecular(0.5);
    volumeProperty->SetAmbient(0.5);

    isoValues = volumeProperty->GetIsoSurfaceValues();

    vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
    volume->SetMapper(mapper);
    volume->SetProperty(volumeProperty);

    renderer->AddActor(volume);
}

void VisitationMapRenderer::updateIsovalue()
{
    std::cout << "isovalue set to " << isovalue << std::endl;
    opacity->RemoveAllPoints();
    opacity->AddPoint(isovalue, SURFACE_TRANSPARENCY);

    isoValues->SetValue(0, isovalue);
}

void VisitationMapRenderer::KeyPressed(const std::basic_string<char> &key)
{
    if(key == "u")
    {
        //Increasing isovalue
        if (isovalue != UINT_MAX)
        {
            isovalue++;
        }
        updateIsovalue();
    }
    else if(key == "j")
    {
        //Decreasing isovalue
        if (isovalue != 0)
        {
            isovalue--;
        }
        updateIsovalue();
    }
    else if(key == "s")
    {
        //Toggle hull smoothing
        if(isSmooth)
        {
            volumeProperty->SetInterpolationTypeToNearest();
        }
        else
        {
            volumeProperty->SetInterpolationTypeToLinear();
        }
        isSmooth = !isSmooth;
    }
}