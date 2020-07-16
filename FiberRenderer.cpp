//
// Created by rutger on 7/2/20.
//

#include <vtkPolyLine.h>
#include "FiberRenderer.h"

FiberRenderer::FiberRenderer(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkRenderWindow> renderWindow)
    : points(vtkSmartPointer<vtkPoints>::New()),
      ids(vtkSmartPointer<vtkIdList>::New()),
      polyLines(vtkSmartPointer<vtkCellArray>::New()),
      polyData(vtkSmartPointer<vtkPolyData>::New()),
      mapper(vtkSmartPointer<vtkPolyDataMapper>::New()),
      actor(vtkSmartPointer<vtkActor>::New()),
      renderer(renderer),
      renderWindow(renderWindow)
{
    initialize();
}

void FiberRenderer::initialize()
{
    polyData->SetPoints(points);
    polyData->SetLines(polyLines);
    mapper->SetInputData(polyData);
    
    actor->SetMapper(mapper);
    
    renderer->AddActor(actor);
}

void FiberRenderer::NewFiber(const Fiber& fiber)
{
    //std::cout << "Line has " << idList->GetNumberOfIds() << " points." << std::endl;
    
    vtkSmartPointer<vtkPolyLine> polyLine = fiber.CreatePolyLine();
    vtkSmartPointer<vtkPoints> fiberPoints = fiber.GetPoints();

    points->InsertPoints(points->GetNumberOfPoints(), fiberPoints->GetNumberOfPoints(), 0, fiberPoints);
    
    // Create a cell array to store the lines in and add the lines to it
    polyLines->InsertNextCell(polyLine);
    
    points->Modified();
    polyLines->Modified();
    renderWindow->Render();
//    polyLines->InsertNextCell(idList);
//    polyLines->Modified();
//    renderWindow->Render();
//
//
//    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
//    polyData->SetPoints(points);
//    polyData->SetVerts(vertices);
    
    std::cout << "Rendered line!" << std::endl;
}
