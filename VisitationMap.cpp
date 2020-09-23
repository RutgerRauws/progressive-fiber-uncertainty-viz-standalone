//
// Created by rutger on 7/20/20.
//

#include "VisitationMap.h"
#include "Point.h"
#include "Cell.h"
#include <vtkPointData.h>

VisitationMap::VisitationMap(double cellSize)
    : cellSize(cellSize),
      vtkData(vtkSmartPointer<vtkPolyData>::New()),
      splatter(vtkSmartPointer<vtkGaussianSplatter>::New()),
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

    splatter->SetInputData(vtkData);
//    splatter->SetSampleDimensions(20, 20, 20); //Higher values produce better results but are much slower.
//    splatter->SetRadius(0.05f); //This value is expressed as a percentage of the length of the longest side of the sampling volume. Smaller numbers greatly reduce execution time.
//    splatter->SetExponentFactor(-10); //sharpness of decay of the splats. This is the exponent constant in the Gaussian equation. Normally this is a negative value.
//    splatter->SetEccentricity(2); //Control the shape of elliptical splatting. Eccentricity > 1 creates needles with the long axis in the direction of the normal; Eccentricity<1 creates pancakes perpendicular to the normal vector.


    splatter->SetSampleDimensions(50, 50, 50); //Higher values produce better results but are much slower.
    splatter->SetRadius(0.1f); //This value is expressed as a percentage of the length of the longest side of the sampling volume. Smaller numbers greatly reduce execution time.
    splatter->SetExponentFactor(-5); //sharpness of decay of the splats. This is the exponent constant in the Gaussian equation. Normally this is a negative value.
    splatter->SetEccentricity(1); //Control the shape of elliptical splatting. Eccentricity > 1 creates needles with the long axis in the direction of the normal; Eccentricity<1 creates pancakes perpendicular to the normal vector.
    splatter->NormalWarpingOff();
//    splatter->ScalarWarpingOff();
    splatter->Update();

//    data = new Cell*[GetNumberOfCells()];
}

void VisitationMap::cellModifiedCallback()
{
    vtkData->Modified();
}

void VisitationMap::InsertPoint(const Point& point) const
{
    vtkIdType cellPtId = vtkData->FindPoint(point.X, point.Y, point.Z);

    if(cellPtId != -1)
    {
        double* cellPt = vtkData->GetPoint(cellPtId);

        if(isInCell(cellPt, point, cellSize))
        {
            frequencies->SetValue(cellPtId, frequencies->GetValue(cellPtId) + 1);
            frequencies->Modified();
            return;
        }
    }

//    double shifted_x = point.X - std::fmod(point.X, cellSize) / 2.0f;
//    double shifted_y = point.Y - std::fmod(point.Y, cellSize) / 2.0f;
//    double shifted_z = point.Z - std::fmod(point.Z, cellSize) / 2.0f;
    double halfSize = cellSize / 2.0f;
    double shifted_x = std::floor(point.X/cellSize) * cellSize + halfSize;
    double shifted_y = std::floor(point.Y/cellSize) * cellSize + halfSize;
    double shifted_z = std::floor(point.Z/cellSize) * cellSize + halfSize;

//    vtkIdType ptId = vtkData->GetPoints()->InsertNextPoint(shifted_x, shifted_y, shifted_z);
    //frequencies->SetValue(ptId, 1);
    vtkData->GetPoints()->InsertNextPoint(shifted_x, shifted_y, shifted_z);
    frequencies->InsertNextValue(1);

    vtkData->GetPoints()->Modified();
    frequencies->Modified();
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
