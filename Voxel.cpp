//
// Created by rutger on 7/20/20.
//

#include "Voxel.h"

Voxel::Voxel()
    : Voxel(Point(0, 0, 0), -1, -1)
{}

Voxel::Voxel(Point position, double size, int value)
        : position(position),
          size(size),
          cubeSource(vtkSmartPointer<vtkCubeSource>::New())
{
    SetValue(value);
    updateVTKObject();
}

vtkSmartPointer<vtkCubeSource> Voxel::GetVTKObject() const
{
    return cubeSource;
}

int Voxel::GetValue() const
{
    return value;
}

void Voxel::SetValue(int val)
{
    if(val < 0)
    {
        throw std::logic_error("Cannot set negative fiber frequency!");
    }

    value = val;
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