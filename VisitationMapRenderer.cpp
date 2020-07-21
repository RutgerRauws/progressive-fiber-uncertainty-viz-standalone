//
// Created by rutger on 7/20/20.
//

#include "VisitationMapRenderer.h"
#include <vtkProperty.h>
#include <vtkAppendPolyData.h>

VisitationMapRenderer::VisitationMapRenderer(VisitationMap& visitationMap,
                                             vtkSmartPointer<vtkRenderer> renderer,
                                             vtkSmartPointer<vtkRenderWindow> renderWindow)
    : visitationMap(visitationMap),
      renderWindow(std::move(renderWindow))
{
    std::cout << "Initializing visitation map renderer... " << std::flush;

    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

    for(unsigned int i = 0; i < visitationMap.GetNumberOfCells(); i++)
    {
        Voxel* voxel = visitationMap.GetCell(i);
        //mapper->AddInputConnection(voxel->GetVTKObject()->GetOutputPort());

        vtkSmartPointer<vtkPolyData> input = vtkSmartPointer<vtkPolyData>::New();
        voxel->GetVTKObject()->Update();
        input->ShallowCopy(voxel->GetVTKObject()->GetOutput());
        appendFilter->AddInputData(input);
    }

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetRepresentationToWireframe();

    renderer->AddActor(actor);

    std::cout << "Complete." << std::endl;
}

void VisitationMapRenderer::NewFiber(const Fiber &fiber)
{
//    for(const Point& point : fiber.GetPoints())
//    {
//        visitationMap.grid->GetPoints()->InsertNextPoint(point.X, point.Y, point.Z);
//    }

//    std::cout << visitationMap.grid->GetDimensions()[0] << ", " << visitationMap.grid->GetDimensions()[1] << ", " << visitationMap.grid->GetDimensions()[2] << std::endl;

    //visitationMap.grid->GetPoints()->Modified();
    //visitationMap.grid->Modified();

    //renderWindow->Render();
}
