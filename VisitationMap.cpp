//
// Created by rutger on 7/20/20.
//

#include "VisitationMap.h"
#include "Point.h"
#include "Voxel.h"

VisitationMap::VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
    : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax)
{
    //TODO: Look into fixing double to int conversion.
    width =  std::abs((xmin - xmax) / voxelSize);
    height = std::abs((ymin - ymax) / voxelSize);
    depth =  std::abs((zmin - zmax) / voxelSize);

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
    for(unsigned int i = 0; i < GetNumberOfCells(); i++)
    {
        delete GetCell(i);
    }

    delete[] data;
}

void VisitationMap::initialize()
{
    std::cout << "Initializing visitation map... " << std::flush;
    data = new Voxel*[GetNumberOfCells()];

    double halfSize = voxelSize / 2.0f;

    int i = 0;

    for(unsigned int x = 0; x < width; x++)
    {
        for(unsigned int y = 0; y < height; y++)
        {
            for(unsigned int z = 0; z < depth; z++)
            {
                //int x = std::floor(xmin + halfSize); x < std::ceil(xmax - halfSize); x = std::floor(x + voxelSize)
                double pos_x = xmin + halfSize + x * voxelSize;
                double pos_y = ymin + halfSize + y * voxelSize;
                double pos_z = zmin + halfSize + z * voxelSize;

                data[x + width * (y + z * height)] = new Voxel(
                    Point(pos_x, pos_y, pos_z),
                    voxelSize,
                    0
                );
            }
        }
    }

    std::cout << "Complete." << std::endl;
}

Voxel* VisitationMap::GetCell(unsigned int x_index, unsigned int y_index, unsigned int z_index) const
{
    //Simplification of x + y * width + z * width * height
    return data[x_index + width * (y_index + z_index * height)];
}

Voxel* VisitationMap::GetCell(unsigned int index) const
{
    return data[index];
}

Voxel* VisitationMap::FindCell(const Point& point) const
{
    if(point.X < xmin || point.X > xmax || point.Y < ymin || point.Y > ymax || point.Z < zmin || point.Z > zmax) {
        return nullptr;
    }

    //TODO: Make use of acceleration data structures such as octrees.
    for(unsigned int x_index = 0; x_index < width; x_index++)
    {
        for (unsigned int y_index = 0; y_index < height; y_index++)
        {
            for (unsigned int z_index = 0; z_index < depth; z_index++)
            {
                Voxel* voxel = GetCell(x_index, y_index, z_index);

                if(voxel->Contains(point))
                {
                    return voxel;
                }
            }
        }
    }

    return nullptr;
}

Voxel* VisitationMap::FindCell(double x, double y, double z) const
{
    return FindCell(Point(x, y, z));
}

void VisitationMap::SetCell(int x, int y, int z, int value)
{
    //Simplification of x + y * WIDTH + z * WIDTH * DEPTH
    data[x + width * (y + z * height)]->SetValue(value);
}

void VisitationMap::SetCell(const Point& point, int value)
{
    SetCell(point.X, point.Y, point.Z, value);
}

unsigned int VisitationMap::GetNumberOfCells() const {
    return width * height * depth;
}

vtkSmartPointer<vtkImageData> VisitationMap::GenerateImageData() const
{
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(width, height, depth);
//    imageData->SetScalarType();
//    imageData->SetNumberOfScalarComponents(1);
//    imageData->SetSpacing();
//    imageData->SetOrigin();
    imageData->AllocateScalars(VTK_UNSIGNED_INT, 1);


    // Fill every entry of the image data with a color
    int* dims = imageData->GetDimensions();
    auto *ptr = static_cast<unsigned int*>(imageData->GetScalarPointer(0, 0, 0));

    for (int z=0; z<dims[2]; z++)
    {
        for (int y = 0; y < dims[1]; y++)
        {
            for (int x = 0; x < dims[0]; x++)
            {
                *ptr++ = GetCell(x, y, z)->GetValue();
            }
        }
    }

    return imageData;
}
