//
// Created by rutger on 8/1/20.
//

#include <string>
#include <vtkPolyDataMapper.h>
#include <vtkPoints.h>
#include <vtkLineSource.h>
#include <vtkPolyLine.h>
#include <vtkTubeFilter.h>
#include <vtkRenderer.h>
#include <vtkProperty.h>
#include "CenterlineRenderer.h"

CenterlineRenderer::CenterlineRenderer(vtkSmartPointer<vtkRenderer> renderer)
    : renderer(std::move(renderer)),
      actor(vtkSmartPointer<vtkActor>::New()),
      centerlineShown(false),
      centerFiberId(std::numeric_limits<unsigned int>::max())
{}

void CenterlineRenderer::NewFiber(Fiber* newCenterline)
{
    distanceTable.InsertNewFiber(*newCenterline);

    if(distanceTable.GetCenterline().GetId() != centerFiberId)
    {
        centerFiberId = distanceTable.GetCenterline().GetId();
        render();
    }
}

void CenterlineRenderer::KeyPressed(const std::basic_string<char>& key)
{
    if(key == "c")
    {
        if(actor != nullptr)
        {
            if (centerlineShown)
            {
                renderer->RemoveActor(actor);
                std::cout << "Centerline hidden" << std::endl;
            }
            else
            {
                renderer->AddActor(actor);
                std::cout << "Centerline shown" << std::endl;
            }
        }

        centerlineShown = !centerlineShown;
    }
}

void CenterlineRenderer::render()
{
    vtkNew<vtkPoints> points;
    vtkNew<vtkCellArray> lines;

    const std::vector<Point>& newPoints = distanceTable.GetCenterline().GetPoints();
    points->SetNumberOfPoints(newPoints.size());

    vtkNew<vtkPolyLine> polyLine;
    polyLine->GetPointIds()->SetNumberOfIds(newPoints.size());

    for(int i = 0; i < newPoints.size(); i++)
    {
        const Point& point = newPoints[i];

        points->InsertPoint(i, point.X, point.Y, point.Z);
        polyLine->GetPointIds()->SetId(i, i);
    }

    lines->InsertNextCell(polyLine);

    vtkNew<vtkPolyData> polyData;
    polyData->SetPoints(points);
    polyData->SetLines(lines);

    vtkNew<vtkPolyDataMapper> polyMapper;
    polyMapper->SetInputData(polyData);

    // Create a tube (cylinder) around the line
    vtkNew<vtkTubeFilter> tubeFilter;
    tubeFilter->SetInputData(polyData);
    tubeFilter->SetCapping(true);
    tubeFilter->SetRadius(.25); //default is .5
    tubeFilter->SetNumberOfSides(6);
    tubeFilter->Update();

    // Create a mapper and actor
    vtkNew<vtkPolyDataMapper> tubeMapper;
    tubeMapper->SetInputConnection(tubeFilter->GetOutputPort());

    if(centerlineShown && actor != nullptr)
    {
        renderer->RemoveActor(actor);
        //actor->Delete();
    }

    actor = vtkSmartPointer<vtkActor>::New();
    actor->GetProperty()->SetColor(1, 1, 1);
    actor->GetProperty()->SetOpacity(0.75);
    actor->SetMapper(tubeMapper);

    if(centerlineShown)
    {
        renderer->AddActor(actor);
        std::cout << "Rendered new centerline" << std::endl;
    }
}
