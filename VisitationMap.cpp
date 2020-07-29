//
// Created by rutger on 7/20/20.
//

#include "VisitationMap.h"
#include "Point.h"
#include "Voxel.h"

VisitationMap::VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
    : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax),
      imageData(vtkSmartPointer<vtkImageData>::New())
{
    //TODO: Look into fixing double to int conversion.
    width =  std::ceil( std::abs(xmin - xmax) / voxelSize);
    height = std::ceil(std::abs(ymin - ymax) / voxelSize);
    depth =  std::ceil(std::abs(zmin - zmax) / voxelSize);

    initialize();
}

/**
 *
 * @param bounds xmin,xmax, ymin,ymax, zmin,zmax
 */
VisitationMap::VisitationMap(double* bounds)
    : VisitationMap(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
{}

VisitationMap::~VisitationMap()
{
    for(int i = 0; i < GetNumberOfCells(); i++)
    {
        delete data[i];
    }

    delete[] data;
}

void VisitationMap::initialize()
{
    std::cout << "Initializing visitation map... " << std::flush;
    data = new Voxel*[GetNumberOfCells()];
    imageData->SetDimensions(width, height, depth);
    imageData->SetSpacing(voxelSize, voxelSize, voxelSize);
    imageData->SetOrigin(xmin, ymin, zmin);

    //Apparently the vtkVolumeRayCastMapper class only works with unsigned char and unsigned short data
    imageData->AllocateScalars(VTK_UNSIGNED_INT, 1);

    // Fill every entry of the image data with a color
    start_ptr = static_cast<unsigned int*>(imageData->GetScalarPointer(0, 0, 0));
    auto* ptr = start_ptr;

    double halfSize = voxelSize / 2.0f;
    for (unsigned int z = 0; z < depth; z++)
    {
        for (unsigned int y = 0; y < height; y++)
        {
            for(unsigned int x = 0; x < width; x++)
            {
                double pos_x = xmin + halfSize + x * voxelSize;
                double pos_y = ymin + halfSize + y * voxelSize;
                double pos_z = zmin + halfSize + z * voxelSize;

                data[x + width * (y + z * height)] = new Voxel(
                        this,
                        Point(pos_x, pos_y, pos_z),
                        voxelSize,
                        ptr
                );

                *ptr++ = 0;
            }
        }
    }

    std::cout << "Complete." << std::endl;
}

Voxel* VisitationMap::GetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index) const
{
    //Simplification of x + y * width + z * width * height
    //return start_ptr + x_index + width * (y_index + z_index * height);
    return data[x_index + width * (y_index + z_index * height)];
}

Voxel* VisitationMap::GetCell(unsigned int index) const
{
    return data[index];
}

Voxel* VisitationMap::FindCell(const Point& point) const
{
    if(point.X < xmin || point.X > xmax || point.Y < ymin || point.Y > ymax || point.Z < zmin || point.Z > zmax)
    {
        std::cerr << "Requested point (" << point.X << ", " << point.Y << ", " << point.Z << " is outside visitation map bounds." << std::endl;
        return nullptr;
    }

    //TODO: Make use of acceleration data structures such as octrees.
    for(unsigned int z_index = 0; z_index < depth + 1; z_index++)
    {
        for(unsigned int y_index = 0; y_index < height + 1; y_index++)
        {
            for(unsigned int x_index = 0; x_index < width + 1; x_index++)
            {
                Voxel* voxel = GetCell(x_index, y_index, z_index);

                if(voxel->Contains(point))
                {
                    return voxel;
                }
            }
        }
    }

    std::cerr << "Requested point (" << point.X << ", " << point.Y << ", " << point.Z << " is not found in voxels of visitation maps." << std::endl;
    return nullptr;
}

Voxel* VisitationMap::FindCell(double x, double y, double z) const
{
    return FindCell(Point(x, y, z));
}

void VisitationMap::SetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index, unsigned int value)
{
//    *getCell(x_index, y_index, z_index) = value;

    //Simplification of x + y * WIDTH + z * WIDTH * DEPTH
    //data[x_index + width * (y_index + z_index * height)]->SetValue(value);

    GetCell(x_index, y_index, z_index)->SetValue(value);
    imageData->Modified();
}

void VisitationMap::SetCell(const Point& point, unsigned int value)
{
//    unsigned int* cell = findCell(point);
//
//    if(cell == nullptr)
//    {
//        std::cerr << "No voxel found at (" << point.X << ", " << point.Y << ", " << point.Z << ") to set value!" << std::endl;
//        return;
//    }
//
//    *cell = value;
    SetCell(point.X, point.Y, point.Z, value);
}

unsigned int VisitationMap::GetNumberOfCells() const
{
    return width * height * depth;
}

vtkSmartPointer<vtkImageData> VisitationMap::GetImageData() const
{
    return imageData;
}

double VisitationMap::GetVoxelSize() const
{
    return voxelSize;
}

void VisitationMap::Modified()
{
    imageData->Modified();
}