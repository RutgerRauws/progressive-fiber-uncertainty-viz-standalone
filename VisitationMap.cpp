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
      splatter(vtkSmartPointer<GaussianFiberSplatter>::New()),
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
    splatter->SetExistingPointsFiberData(fibers);
    //splatter->SetSampleDimensions(80, 80, 80); //Higher values produce better results but are much slower.
    splatter->SetRadius(0.05f); //This value is expressed as a percentage of the length of the longest side of the sampling volume. Smaller numbers greatly reduce execution time.
//    splatter->SetRadius(0.01f);

//    splatter->SetExponentFactor(-10); //sharpness of decay of the splats. This is the exponent constant in the Gaussian equation. Normally this is a negative value.
//    splatter->SetEccentricity(2); //Control the shape of elliptical splatting. Eccentricity > 1 creates needles with the long axis in the direction of the normal; Eccentricity<1 creates pancakes perpendicular to the normal vector.

//                            int sampleDimension = 40; //indicates quality
//                            splatter->SetSampleDimensions(sampleDimension, sampleDimension, sampleDimension); //Higher values produce better results but are much slower.
//
//                            double radius =  ((float)sampleDimension / 2.0f) / cellSize;
//                            double ratio = 0.1f * radius / ((float)sampleDimension);
//                            splatter->SetRadius(ratio);

//    int sampleDimension = 20; //indicates quality
//    splatter->SetSampleDimensions(sampleDimension, sampleDimension, sampleDimension); //Higher values produce better results but are much slower.

    //double radius =  ((float)sampleDimension / 2.0f) * cellSize;
    //double ratio = radius / (sampleDimension * cellSize);
//    double ratio = 1.0f / ((float)sampleDimension);
//    splatter->SetRadius(ratio);

//    splatter->SetRadius(0.05f); //This value is expressed as a percentage of the length of the longest side of the sampling volume. Smaller numbers greatly reduce execution time.

    splatter->SetExponentFactor(-5); //sharpness of decay of the splats. This is the exponent constant in the Gaussian equation. Normally this is a negative value.
    splatter->SetEccentricity(1); //Control the shape of elliptical splatting. Eccentricity > 1 creates needles with the long axis in the direction of the normal; Eccentricity<1 creates pancakes perpendicular to the normal vector.
    //splatter->NormalWarpingOff();
//    splatter->ScalarWarpingOff();
    splatter->Update();
}

void VisitationMap::InsertFiber(const Fiber& fiber)
{
    //TODO: This should be based on edges
    for(const Point& point : fiber.GetPoints())
    {
        insertPoint(point, fiber);
    }

    frequencies->Modified();
    vtkData->GetPoints()->Modified();
//    updateBounds();
}

void VisitationMap::insertPoint(const Point& point, const Fiber& fiber)
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

void VisitationMap::updateBounds()
{
    double bounds[6];
    vtkData->GetBounds(bounds);

    double width  = std::abs(bounds[0] - bounds[1]);
    double height = std::abs(bounds[2] - bounds[3]);
    double depth  = std::abs(bounds[4] - bounds[5]);

    std::cout << width << ", " << height << ", " << depth << std::endl;

    int numberOfCellsWidth = std::ceil(width / cellSize);
    int numberOfCellsHeight = std::ceil(height / cellSize);
    int numberOfCellsDepth = std::ceil(depth / cellSize);

    splatter->SetSampleDimensions(numberOfCellsWidth, numberOfCellsHeight, numberOfCellsDepth); //Higher values produce better results but are much slower.

    //The radius is expressed as a percentage of the length of the longest side of the sampling volume.
    //Smaller numbers greatly reduce execution time.

    double longestSide = std::max(width, std::max(height, depth));

    double radius =  cellSize / longestSide;

    std::cout << radius << std::endl;

    splatter->SetRadius(radius);

    splatter->Update();
}
