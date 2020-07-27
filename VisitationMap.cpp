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

void VisitationMap::initialize()
{
    std::cout << "Initializing visitation map... " << std::flush;
    imageData->SetDimensions(width, height, depth);
    imageData->SetSpacing(voxelSize, voxelSize, voxelSize);
    imageData->SetOrigin(xmin, ymin, zmin);

    //Apparently the vtkVolumeRayCastMapper class only works with unsigned char and unsigned short data
    imageData->AllocateScalars(VTK_UNSIGNED_INT, 1);

    // Fill every entry of the image data with a color
    start_ptr = static_cast<unsigned int*>(imageData->GetScalarPointer(0, 0, 0));
    auto* ptr = start_ptr;

    for(unsigned int i = 0; i < GetNumberOfCells(); i++)
    {
        *ptr++ = 0;
    }

    std::cout << "Complete." << std::endl;
}

unsigned int* VisitationMap::getCell(unsigned int x_index, unsigned int y_index, unsigned int z_index) const
{
    //Simplification of x + y * width + z * width * height
    return start_ptr + x_index + width * (y_index + z_index * height);
}

unsigned int *VisitationMap::getCell(unsigned int index) const
{
    return start_ptr + index;
}

unsigned int* VisitationMap::findCell(const Point& point) const
{
    if(point.X < xmin || point.X > xmax || point.Y < ymin || point.Y > ymax || point.Z < zmin || point.Z > zmax)
    {
        return nullptr;
    }

    //TODO: Make use of acceleration data structures such as octrees.
    for(unsigned int z_index = 0; z_index < depth + 1; z_index++)
    {
        for(unsigned int y_index = 0; y_index < height + 1; y_index++)
        {
            for(unsigned int x_index = 0; x_index < width + 1; x_index++)
            {
                if(containedInCell(x_index, y_index, z_index, point))
                {
                    return getCell(x_index, y_index, z_index);
                }
            }
        }
    }

    return nullptr;
}

unsigned int VisitationMap::GetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index) const
{
    return *getCell(x_index, y_index, z_index);
}

unsigned int VisitationMap::GetCell(unsigned int index) const
{
    return *getCell(index);
}

unsigned int VisitationMap::FindCell(const Point& point) const
{
    unsigned int* cell = findCell(point);

    if(cell == nullptr)
    {
        std::cerr << "Requested point (" << point.X << ", " << point.Y << ", " << point.Z << " is not found in voxels of visitation maps." << std::endl;
        return 0;
    }

    return *cell;
}

unsigned int VisitationMap::FindCell(double x, double y, double z) const
{
    return FindCell(Point(x, y, z));
}

void VisitationMap::SetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index, unsigned int value)
{
    *getCell(x_index, y_index, z_index) = value;
    imageData->Modified();
}

void VisitationMap::SetCell(const Point& point, unsigned int value)
{
    unsigned int* cell = findCell(point);

    if(cell == nullptr)
    {
        std::cerr << "No voxel found at (" << point.X << ", " << point.Y << ", " << point.Z << ") to set value!" << std::endl;
        return;
    }

    *cell = value;
    imageData->Modified();
}

unsigned int VisitationMap::GetNumberOfCells() const
{
    return width * height * depth;
}

vtkSmartPointer<vtkImageData> VisitationMap::GetImageData() const
{
    return imageData;
}

bool VisitationMap::containedInCell(unsigned int x_index, unsigned int y_index, unsigned int z_index,
                                    const Point &point) const
{
    double halfSize = voxelSize / 2.0f;

    //Cell center
    double pos_x = xmin + halfSize + x_index * voxelSize;
    double pos_y = ymin + halfSize + y_index * voxelSize;
    double pos_z = zmin + halfSize + z_index * voxelSize;

    double voxel_xmin = pos_x - halfSize;
    double voxel_xmax = pos_x + halfSize;
    double voxel_ymin = pos_y - halfSize;
    double voxel_ymax = pos_y + halfSize;
    double voxel_zmin = pos_z - halfSize;
    double voxel_zmax = pos_z + halfSize;

    return (voxel_xmin <= point.X) && (point.X <= voxel_xmax)
           && (voxel_ymin <= point.Y) && (point.Y <= voxel_ymax)
           && (voxel_zmin <= point.Z) && (point.Z <= voxel_zmax);
}

