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
#include <vtkImageGaussianSmooth.h>

VisitationMapRenderer::VisitationMapRenderer(VisitationMap &visitationMap,
                                             vtkSmartPointer<vtkRenderer> renderer,
                                             int startIsovalue,
                                             bool isSmooth)
    : visitationMap(visitationMap),
      renderer(std::move(renderer)),
      actor(vtkSmartPointer<vtkActor>::New()),
      smoother(vtkSmartPointer<vtkImageGaussianSmooth>::New()),
      mapper(vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper>::New()),
      isovalue(startIsovalue),
      isSmooth(isSmooth)
{
    initialize();
}

void VisitationMapRenderer::initialize()
{
    isovalue++;
    //vtkSmartPointer<vtkVolumeRayCastIsosurfaceFunction> b = vtkSmartPointer<vtkVolumeRayCastIsosurfaceFunction>::New();
    //vtkSmartPointer<vtkGPUVolumeRayCastMapper> mapper = vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();

//    mapper->SetInputData(visitationMap.GetImageData());
//    mapper->SetInputData(visitationMap.GetOutput());
    mapper->SetInputConnection(visitationMap.GetOutputPort());
//    mapper->SetInputData(visitationMap.GetOutput());
//
//    visitationMap.Update();

    smoother->SetDimensionality(3);
//    smoother->SetInputConnection(visitationMap.GetOutputPort());
//    smoother->SetInputData(visitationMap.GetOutput());
    smoother->SetStandardDeviations(4, 4, 4);
    smoother->SetRadiusFactors(3, 3, 3);


//    mapper->SetInputConnection(smoother->GetOutputPort());
//    mapper->SetPartitions(2,2,1);
    mapper->AutoAdjustSampleDistancesOff();
//    mapper->SetAutoAdjustSampleDistances(true);
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