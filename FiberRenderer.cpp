//
// Created by rutger on 7/2/20.
//

#include <vtkPolyLine.h>
#include <vtkProperty.h>
#include "FiberRenderer.h"

FiberRenderer::FiberRenderer(vtkSmartPointer<vtkRenderer> renderer)
    : renderer(std::move(renderer)),
      currentId(0),
      points(vtkSmartPointer<vtkPoints>::New()),
      polyLines(vtkSmartPointer<vtkCellArray>::New()),
      fiberActor(vtkSmartPointer<vtkActor>::New()),
      pointsActor(vtkSmartPointer<vtkActor>::New()),
      vertexGlyphFilter(vtkSmartPointer<vtkVertexGlyphFilter>::New()),
      fibersShown(true),
      pointsShown(false)
{
    initialize();
}

void FiberRenderer::initialize()
{
    /*
     * Fibers
     */
    vtkNew<vtkPolyData> polyData;
    polyData->SetPoints(points);
    polyData->SetLines(polyLines);

    vertexGlyphFilter->AddInputData(polyData);
    vertexGlyphFilter->Update();

    vtkNew<vtkPolyDataMapper> fiberMapper;
    fiberMapper->SetInputData(polyData);

    fiberActor->SetMapper(fiberMapper);
    fiberActor->GetProperty()->SetLineWidth(2);

    if(fibersShown)
    {
        renderer->AddActor(fiberActor);
    }

    /*
     * Points
     */
    auto glyphMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    glyphMapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());

    pointsActor->SetMapper(glyphMapper);
    pointsActor->GetProperty()->SetColor(1, 0, 0);
    pointsActor->GetProperty()->SetPointSize(5);

    if(pointsShown)
    {
        renderer->AddActor(pointsActor);
    }
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
}

void FiberRenderer::KeyPressed(const std::basic_string<char>& key)
{
    if(key == "p")
    {
        if(pointsShown)
        {
            renderer->RemoveActor(pointsActor);
        }
        else
        {
            renderer->AddActor(pointsActor);
        }

        pointsShown = !pointsShown;
    }
    else if(key == "f")
    {
        if(fibersShown)
        {
            renderer->RemoveActor(fiberActor);
        }
        else
        {
            renderer->AddActor(fiberActor);
        }

        fibersShown = !fibersShown;
    }
}