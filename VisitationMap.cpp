//
// Created by rutger on 7/20/20.
//

#include "VisitationMap.h"
#include "Point.h"
#include "Cell.h"
#include "GaussianFiberSplatter.h"
#include <vtkPointData.h>

VisitationMap::VisitationMap(double cellSize)
    : cellSize(cellSize),
      vtkData(vtkSmartPointer<vtkPolyData>::New()),
//      splatter(vtkSmartPointer<GaussianFiberSplatter>::New()),
      splatter(vtkSmartPointer<FiberSplatter>::New()),
      frequencies(vtkSmartPointer<vtkUnsignedIntArray>::New()),
      distanceScores(vtkSmartPointer<vtkDoubleArray>::New())
{
    initialize();
}

void VisitationMap::initialize()
{
    std::cout << "Initializing visitation map... " << std::flush;
    frequencies->SetName("FiberFrequencies");
    distanceScores->SetName("DistanceScores");

    vtkData->SetPoints(vtkSmartPointer<vtkPoints>::New());

    vtkData->GetPointData()->AddArray(frequencies);
    vtkData->GetPointData()->AddArray(distanceScores);

    vtkData->GetPointData()->SetActiveScalars(frequencies->GetName());

    splatter->CellSize = cellSize;
    splatter->KernelRadius = 1;

    splatter->SetInputData(vtkData);
    splatter->SetCellFrequencies(frequencies);
    splatter->SetExistingPointsFiberData(fibers);

    splatter->Update();
}

//void VisitationMap::InsertFiber(const Fiber& fiber)
//{
//    //TODO: This should be based on edges
//    for(const Point& point : fiber.GetPoints())
//    {
//        insertPoint(point, fiber);
//    }
//
//    frequencies->Modified();
//    vtkData->GetPoints()->Modified();
////    updateBounds();
//}

void VisitationMap::InsertPoint(const Point& point, const Fiber& fiber)
{
    vtkIdType cellPtId = vtkData->FindPoint(point.X, point.Y, point.Z);

    if(cellPtId != -1)
    {
        double* cellPt = vtkData->GetPoint(cellPtId);

        if(isInCell(cellPt, point, cellSize))
        {
            fibers[cellPtId].emplace_back(fiber);

            frequencies->SetValue(cellPtId, frequencies->GetValue(cellPtId) + 1);
            return;
        }
    }

    double halfSize = cellSize / 2.0f;
    double shifted_x = std::floor(point.X/cellSize) * cellSize + halfSize;
    double shifted_y = std::floor(point.Y/cellSize) * cellSize + halfSize;
    double shifted_z = std::floor(point.Z/cellSize) * cellSize + halfSize;

    vtkData->GetPoints()->InsertNextPoint(shifted_x, shifted_y, shifted_z);
    frequencies->InsertNextValue(1);
    fibers.push_back({fiber});
}

void VisitationMap::InsertSphere(const Point& point, const Fiber& fiber, double radius)
{
    double halfSize = cellSize / 2.0f;
    double shifted_x = std::floor(point.X/cellSize) * cellSize + halfSize;
    double shifted_y = std::floor(point.Y/cellSize) * cellSize + halfSize;
    double shifted_z = std::floor(point.Z/cellSize) * cellSize + halfSize;

    Point centerPoint(shifted_x, shifted_y, shifted_z);
    InsertPoint(centerPoint, fiber);


    //TODO: is this correct?
    int radiusInNumberCells = std::ceil(radius / GetCellSize());

    for(int x_index_offset = -radiusInNumberCells; x_index_offset < radiusInNumberCells; x_index_offset++)
    {
        for(int y_index_offset = -radiusInNumberCells; y_index_offset < radiusInNumberCells; y_index_offset++)
        {
            for(int z_index_offset = -radiusInNumberCells; z_index_offset < radiusInNumberCells; z_index_offset++)
            {
                Point splatPoint(
                    shifted_x + x_index_offset * GetCellSize(),
                    shifted_y + y_index_offset * GetCellSize(),
                    shifted_z + z_index_offset * GetCellSize()
                );

                if(isCellInsideSphere(centerPoint, radius, splatPoint, GetCellSize()))
                {
                    InsertPoint(splatPoint, fiber);
                }
            }
        }
    }
}

unsigned int VisitationMap::GetFrequency(const Point& point) const
{
    vtkIdType ptId = vtkData->FindPoint(point.X, point.Y, point.Z);
    unsigned int frequency = frequencies->GetValue(ptId);

    return frequency;
}

double VisitationMap::GetCellSize() const
{
    return cellSize;
}

vtkSmartPointer<vtkPolyData> VisitationMap::GetVTKData() const
{
    return vtkData;
}

vtkAlgorithmOutput* VisitationMap::GetImageOutput() const
{
    return splatter->GetOutputPort();
}

bool VisitationMap::isInCell(const double* cellCenterPoint, const Point& point, double cellSize)
{
    double x = cellCenterPoint[0];
    double y = cellCenterPoint[1];
    double z = cellCenterPoint[2];

    double halfSize = cellSize / 2.0f;

    double xmin = x - halfSize;
    double xmax = x + halfSize;
    double ymin = y - halfSize;
    double ymax = y + halfSize;
    double zmin = z - halfSize;
    double zmax = z + halfSize;

    return (xmin <= point.X) && (point.X <= xmax)
           && (ymin <= point.Y) && (point.Y <= ymax)
           && (zmin <= point.Z) && (point.Z <= zmax);
}


bool VisitationMap::isCellInsideSphere(const Point& center, double radius, const Point& point, double cellSize)
{
    double halfSize = cellSize / 2.0f;

    if(isPointInsideSphere(center, radius, point.X - halfSize, point.Y - halfSize, point.Z - halfSize)
       || isPointInsideSphere(center, radius, point.X - halfSize, point.Y - halfSize, point.Z + halfSize)
       || isPointInsideSphere(center, radius, point.X - halfSize, point.Y + halfSize, point.Z - halfSize)
       || isPointInsideSphere(center, radius, point.X - halfSize, point.Y + halfSize, point.Z + halfSize)
       || isPointInsideSphere(center, radius, point.X + halfSize, point.Y - halfSize, point.Z - halfSize)
       || isPointInsideSphere(center, radius, point.X + halfSize, point.Y - halfSize, point.Z + halfSize)
       || isPointInsideSphere(center, radius, point.X + halfSize, point.Y + halfSize, point.Z - halfSize)
       || isPointInsideSphere(center, radius, point.X + halfSize, point.Y + halfSize, point.Z + halfSize)
            )
    {
        return true;
    }
    else
    {
        return false;
    }
}


bool VisitationMap::isPointInsideSphere(const Point& center, double radius, double x, double y, double z)
{
    return pow(x - center.X, 2)
           +  pow(y - center.Y, 2)
           +  pow(z - center.Z, 2) <= std::pow(radius, 2);
}

void VisitationMap::Modified()
{
    frequencies->Modified();
    vtkData->GetPoints()->Modified();
}
