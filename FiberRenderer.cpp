//
// Created by rutger on 7/2/20.
//

#include <vtkPolyLine.h>
#include <vtkProperty.h>
#include "FiberRenderer.h"

FiberRenderer::FiberRenderer(vtkSmartPointer<vtkRenderer> renderer)
    : currentId(0),
      points(vtkSmartPointer<vtkPoints>::New()),
      polyLines(vtkSmartPointer<vtkCellArray>::New()),
      polyData(vtkSmartPointer<vtkPolyData>::New()),
      mapper(vtkSmartPointer<vtkPolyDataMapper>::New()),
      actor(vtkSmartPointer<vtkActor>::New()),
      renderer(std::move(renderer))
{
    initialize();
}

void FiberRenderer::initialize()
{
    polyData->SetPoints(points);
    polyData->SetLines(polyLines);
    mapper->SetInputData(polyData);
    
    actor->SetMapper(mapper);
    actor->GetProperty()->SetLineWidth(2);
    
    renderer->AddActor(actor);
}

void FiberRenderer::NewFiber(const Fiber& fiber)
{
    const std::vector<Point>& fiberPoints = fiber.GetPoints();

    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
    polyLine->GetPointIds()->SetNumberOfIds(fiberPoints.size());

    for(unsigned int i = 0; i < fiberPoints.size(); i++)
    {
        const Point& point = fiberPoints[i];

        points->InsertPoint(currentId, point.X, point.Y, point.Z);
        polyLine->GetPointIds()->SetId(i, currentId);
        currentId++;
    }

    // Create a cell array to store the lines in and add the lines to it
    polyLines->InsertNextCell(polyLine);

    points->Modified();
    polyData->Modified();
}