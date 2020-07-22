//
// Created by rutger on 7/20/20.
//

#include "VisitationMapRenderer.h"
#include <vtkProperty.h>
#include <vtkAppendPolyData.h>

VisitationMapRenderer::VisitationMapRenderer(VisitationMap& visitationMap,
                                             vtkSmartPointer<vtkRenderer> renderer)
    : visitationMap(visitationMap),
      renderer(renderer),
      actor(vtkSmartPointer<vtkActor>::New())
{
    initialize();
}

void VisitationMapRenderer::initialize()
{
    std::cout << "Initializing visitation map renderer... " << std::flush;

    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

    for(unsigned int i = 0; i < visitationMap.GetNumberOfCells(); i++)
    {
        Voxel* voxel = visitationMap.GetCell(i);

        vtkSmartPointer<vtkPolyData> input = vtkSmartPointer<vtkPolyData>::New();
        voxel->GetVTKObject()->Update();
        input->ShallowCopy(voxel->GetVTKObject()->GetOutput());
        appendFilter->AddInputData(input);
    }

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    actor->SetMapper(mapper);
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->SetRepresentationToWireframe();

    renderer->AddActor(actor);
    std::cout << "Complete." << std::endl;
}


void VisitationMapRenderer::NewFiber(const Fiber &fiber)
{
    //TODO: This should be based on edges
    for(const Point& point : fiber.GetPoints())
    {
        Voxel* voxel = visitationMap.FindCell(point);

        if(voxel == nullptr) {
            continue;
        }

        voxel->SetValue(voxel->GetValue() + 1);
    }

//    std::cout << visitationMap.grid->GetDimensions()[0] << ", " << visitationMap.grid->GetDimensions()[1] << ", " << visitationMap.grid->GetDimensions()[2] << std::endl;

    //visitationMap.grid->GetPoints()->Modified();
    //visitationMap.grid->Modified();

    //renderWindow->Render();
}
