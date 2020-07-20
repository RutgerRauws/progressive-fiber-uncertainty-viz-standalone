//
// Created by rutger on 7/2/20.
//

#include <vtkPolyLine.h>
#include "FiberRenderer.h"

FiberRenderer::FiberRenderer(vtkSmartPointer<vtkRenderer> renderer, vtkSmartPointer<vtkRenderWindow> renderWindow)
    : currentId(0),
      points(vtkSmartPointer<vtkPoints>::New()),
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
    const std::list<Point>& fiberPoints = fiber.GetPoints();

    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
    polyLine->GetPointIds()->SetNumberOfIds(fiberPoints.size());

    for(const Point& point : fiberPoints)
    {
        auto id = points->InsertNextPoint(point.X, point.Y, point.Z);
        polyLine->GetPointIds()->InsertNextId(id);
        currentId++;
    }

    //polyData->GetPoints()->GetData()->Modified(); //this gives SIGSEGV
    //polyData->GetPoints()->Modified(); //this gives SIGSEGV

    // Create a cell array to store the lines in and add the lines to it
    polyLines->InsertNextCell(polyLine);
//    polyData->GetLines()->GetData()->Modified();
    std::cout << points->GetNumberOfPoints() << " | " << polyData->GetPoints()->GetNumberOfPoints() << " | " << currentId << std::endl;
    polyData->Modified();
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
