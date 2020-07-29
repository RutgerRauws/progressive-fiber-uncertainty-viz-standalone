//
// Created by rutger on 7/20/20.
//

#include "Voxel.h"

#include <vtkCellData.h>
#include "VisitationMap.h"

static unsigned int GLOBAL_ID = 0;

Voxel::Voxel(VisitationMap* visitationMap, Point position, double size, unsigned int* value_ptr)
        : id(GLOBAL_ID++),
          visitationMap(visitationMap),
          value_ptr(value_ptr),
          position(position),
          size(size),
          cubeSource(vtkSmartPointer<vtkCubeSource>::New())
{
    if(value_ptr == nullptr)
    {
        std::cerr << "Assigned value pointer is nullptr" << std::endl;
    }
    else
    {
        updateVTKObject();
    }
}

vtkSmartPointer<vtkCubeSource> Voxel::GetVTKObject() const
{
    return cubeSource;
}

unsigned int Voxel::GetValue() const
{
    if(value_ptr == nullptr)
    {
        return 0;
    }

    return *value_ptr;
}

void Voxel::SetValue(unsigned int val)
{
    *value_ptr = val;
    visitationMap->Modified();
    //updateVTKObject();
}

Point Voxel::GetPosition() const
{
    return position;
}

void Voxel::GetBounds(double* bounds) const
{
    double halfSize = size / 2.0f;

    bounds[0] = position.X - halfSize; //xMin
    bounds[1] = position.X + halfSize; //xMax
    bounds[2] = position.Y - halfSize; //yMin
    bounds[3] = position.Y + halfSize; //yMax
    bounds[4] = position.Z - halfSize; //zMin
    bounds[5] = position.Z + halfSize; //zMax
}

bool Voxel::Contains(const Point& point) const
{
    double halfSize = size / 2.0f;
    double xmin = position.X - halfSize;
    double xmax = position.X + halfSize;
    double ymin = position.Y - halfSize;
    double ymax = position.Y + halfSize;
    double zmin = position.Z - halfSize;
    double zmax = position.Z + halfSize;

    return (xmin <= point.X) && (point.X <= xmax)
        && (ymin <= point.Y) && (point.Y <= ymax)
        && (zmin <= point.Z) && (point.Z <= zmax);
}

void Voxel::updateVTKObject()
{
    double halfSize = size / 2.0f;

    cubeSource->SetBounds(
        position.X - halfSize,
        position.X + halfSize,
        position.Y - halfSize,
        position.Y + halfSize,
        position.Z - halfSize,
        position.Z + halfSize
    );

    cubeSource->Update();
}

unsigned int Voxel::GetID() const {
    return id;
}

