//
// Created by rutger on 7/20/20.
//

#include "VisitationMapDebugRenderer.h"
#include <utility>
#include <vtkProperty.h>
#include <vtkAppendPolyData.h>

VisitationMapDebugRenderer::VisitationMapDebugRenderer(VisitationMap& visitationMap,
                                                       vtkSmartPointer<vtkRenderer> renderer)
    : visitationMap(visitationMap),
      renderer(std::move(renderer)),
      actor(vtkSmartPointer<vtkActor>::New())
{
    initialize();
}

void VisitationMapDebugRenderer::initialize()
{
    std::cout << "Initializing visitation map renderer... " << std::flush;

    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

//    for(unsigned int i = 0; i < visitationMap.GetNumberOfCells(); i++)
//    {
//        Cell* voxel = visitationMap.GetCell(i);

//        vtkSmartPointer<vtkPolyData> input = vtkSmartPointer<vtkPolyData>::New();
//        voxel->GetVTKObject()->Update();
//        input->ShallowCopy(voxel->GetVTKObject()->GetOutput());
//        appendFilter->AddInputData(input);
//    }

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    actor->SetMapper(mapper);
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->SetRepresentationToWireframe();

    renderer->AddActor(actor);
    std::cout << "Complete." << std::endl;
}