//
// Created by rutger on 7/23/20.
//

#include "VisitationMapRenderer.h"

#include <utility>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>

VisitationMapRenderer::VisitationMapRenderer(VisitationMap &visitationMap, vtkSmartPointer<vtkRenderer> renderer)
    : visitationMap(visitationMap),
      renderer(std::move(renderer)),
      actor(vtkSmartPointer<vtkActor>::New())
{
    initialize();
}

void VisitationMapRenderer::initialize()
{
    //vtkSmartPointer<vtkVolumeRayCastIsosurfaceFunction> b = vtkSmartPointer<vtkVolumeRayCastIsosurfaceFunction>::New();
    vtkSmartPointer<vtkGPUVolumeRayCastMapper> mapper = vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();
    mapper->SetInputData(visitationMap.GetImageData());

    opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
    NewIsovalue(1);

    vtkSmartPointer<vtkColorTransferFunction> color = vtkSmartPointer<vtkColorTransferFunction>::New();
    color->AddRGBPoint(0, 0, 0, 1);

    vtkSmartPointer<vtkVolumeProperty> property = vtkSmartPointer<vtkVolumeProperty>::New();
    property->SetScalarOpacity(opacity);
    property->SetColor(color);
    property->ShadeOn();
    property->SetDiffuse(0.6);
    property->SetSpecular(0.5);
    property->SetAmbient(0.5);

    vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
    volume->SetMapper(mapper);
    volume->SetProperty(property);

    renderer->AddActor(volume);
}

void VisitationMapRenderer::NewIsovalue(unsigned int value)
{
    std::cout << "isovalue set to " << value << std::endl;

    opacity->RemoveAllPoints();

    opacity->AddPoint(0, 0);
    opacity->AddPoint(value - 1, 0);
    opacity->AddPoint(value, SURFACE_TRANSPARENCY);
}
