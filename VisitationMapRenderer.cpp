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
    mapper->SetInputData(visitationMap.GenerateImageData());

//    vtkSmartPointer<vtkPiecewiseFunction> opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
//    opacity->AddPoint(0, 0);
//    opacity->AddPoint(1, 0.5);
//    opacity->AddPoint(2, 0.5);
//    opacity->AddPoint(3, 0.5);
//    opacity->AddPoint(4, 0.5);
//    opacity->AddPoint(5, 0.5);
//    opacity->AddPoint(6, 0.5);
//    opacity->AddPoint(7, 0.5);
//    opacity->AddPoint(8, 0.5);

    double colors[8][3] = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {1, 1, 0},
            {0, 1, 1},
            {1, 0, 1},
            {0.5, 1, 0.5},
            {1, 1, 0.5} };
    vtkSmartPointer<vtkColorTransferFunction> color = vtkSmartPointer<vtkColorTransferFunction>::New();
    for (int i = 0; i < 8; i++) color->AddRGBPoint(i, colors[i][0], colors[i][1], colors[i][2]);

    vtkSmartPointer<vtkVolumeProperty> property = vtkSmartPointer<vtkVolumeProperty>::New();
    //property->SetScalarOpacity(opacity);
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

