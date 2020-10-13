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

VisitationMapRenderer::VisitationMapRenderer(VisitationMap &visitationMap,
                                             vtkSmartPointer<vtkRenderer> renderer,
                                             int startIsovalue,
                                             bool isSmooth)
    : visitationMap(visitationMap),
      renderer(std::move(renderer)),
      actor(vtkSmartPointer<vtkActor>::New()),
      percentage(0.01f),
      numberOfFibers(0),
      isSmooth(isSmooth)
{
    initialize();
}

void VisitationMapRenderer::initialize()
{
    vtkNew<vtkOpenGLGPUVolumeRayCastMapper> mapper;
    mapper->SetInputData(visitationMap.GetImageData());
    mapper->AutoAdjustSampleDistancesOff();
    mapper->SetSampleDistance(0.01f);

    /*TODO:
     * Some voxels in the vtkImageData are not rendered with SetBelndModeToIsoSurface()
     * They are rendered with the default rendering method, however, so something weird is going on that might need
     * to be reported to the VTK issue tracker
     */
    mapper->SetBlendModeToIsoSurface();

    opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();

    vtkSmartPointer<vtkColorTransferFunction> color = vtkSmartPointer<vtkColorTransferFunction>::New();
    color->AddRGBPoint(0, 1, 1, 0);

    volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
    volumeProperty->SetScalarOpacity(opacity);
    volumeProperty->SetColor(color);
    volumeProperty->ShadeOn();
    volumeProperty->SetDiffuse(0.6);
    volumeProperty->SetSpecular(0.5);
    volumeProperty->SetAmbient(0.5);
    setSmoothing(isSmooth);

    isoValues = volumeProperty->GetIsoSurfaceValues();

    vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
    volume->SetMapper(mapper);
    volume->SetProperty(volumeProperty);

    renderer->AddActor(volume);

    updateIsovalue();
}

void VisitationMapRenderer::updateIsovalue()
{
    float isovalue = ((float)numberOfFibers) * percentage;
    std::cout << "percentage set to " << percentage * 100 << "% which is an isovalue of " << isovalue << std::endl;
    opacity->RemoveAllPoints();
    opacity->AddPoint(isovalue, SURFACE_TRANSPARENCY);
//    opacity->AddPoint(2, 0.2);
//    opacity->AddPoint(isovalue, 0.7);

    isoValues->SetValue(0, isovalue);
//    isoValues->SetValue(0, 2);
//    isoValues->SetValue(1, isovalue);
}

void VisitationMapRenderer::KeyPressed(const sf::Keyboard::Key& key)
{
    if(key == sf::Keyboard::U)
    {
        //Increasing isovalue percentage
        if (percentage + PERCENTAGE_DELTA <= 1.0f)
        {
            percentage += PERCENTAGE_DELTA;
        }
        updateIsovalue();
    }
    else if(key == sf::Keyboard::J)
    {
        //Decreasing isovalue percentage
        if (percentage - PERCENTAGE_DELTA >= 0.0f)
        {
            percentage -= PERCENTAGE_DELTA;
        }
        updateIsovalue();
    }
    else if(key == sf::Keyboard::S)
    {
        //Toggle hull smoothing
        isSmooth = !isSmooth;
        setSmoothing(isSmooth);
    }
}

void VisitationMapRenderer::setSmoothing(bool on)
{
    if(on)
    {
        volumeProperty->SetInterpolationTypeToLinear();
    }
    else
    {
        volumeProperty->SetInterpolationTypeToNearest();
    }
}

void VisitationMapRenderer::NewFiber(Fiber *fiber)
{
    numberOfFibers++;
    updateIsovalue();
}