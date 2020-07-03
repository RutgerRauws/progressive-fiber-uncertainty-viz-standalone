//
// Created by rutger on 7/2/20.
//

#include <vtkPolyData.h>
#include <vtkPolyLine.h>
#include "Fiber.h"

Fiber::Fiber()
    : points(vtkSmartPointer<vtkPoints>::New()),
      ids(vtkSmartPointer<vtkIdList>::New()),
      vertices(vtkSmartPointer<vtkCellArray>::New())
{}

void Fiber::AddPoint(double x, double y, double z)
{
    vtkIdType id = points->InsertNextPoint(x, y, z);
    ids->InsertNextId(id);
}

vtkSmartPointer<vtkPoints> Fiber::GetPoints() const
{
    return points;
}

vtkSmartPointer<vtkIdList> Fiber::GetIds() const
{
    return ids;
}

vtkSmartPointer<vtkPolyLine> Fiber::CreatePolyLine() const
{
    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
    polyLine->GetPointIds()->SetNumberOfIds(points->GetNumberOfPoints());
    
    for(unsigned int i = 0; i < points->GetNumberOfPoints(); i++)
    {
        polyLine->GetPointIds()->SetId(i, i);
    }
    
    return polyLine;
}