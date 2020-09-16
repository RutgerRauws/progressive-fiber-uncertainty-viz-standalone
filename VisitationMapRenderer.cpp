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
      isovalue(startIsovalue),
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
//    opacity->AddPoint(2, 0.2);
//    opacity->AddPoint(isovalue, 0.7);

    isoValues->SetValue(0, isovalue);
//    isoValues->SetValue(0, 2);
//    isoValues->SetValue(1, isovalue);
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
