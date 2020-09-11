//
// Created by rutger on 7/20/20.
//

#include "VisitationMap.h"
#include "Point.h"
#include "Cell.h"

VisitationMap::VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
    : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax),
      imageData(vtkSmartPointer<vtkImageData>::New())
{
    //TODO: Look into fixing double to int conversion.
    width =  std::ceil(std::abs(xmin - xmax) / cellSize);
    height = std::ceil(std::abs(ymin - ymax) / cellSize);
    depth =  std::ceil(std::abs(zmin - zmax) / cellSize);

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

    data = new Cell*[GetNumberOfCells()];

    imageData->SetExtent(0, width - 1, 0, height - 1, 0, depth - 1);

    // TODO: Perhaps the SetSpacing and SetOrigin calls should be removed.
    // As we use vtkMRMLVolumeNode::SetAndObserveImageData, the ImageData object origin must be set to (0,0,0) and
    // spacing must be set to (1,1,1). If the variables are set to different values then the application's behavior is
    // undefined.
    // https://apidocs.slicer.org/v4.8/classvtkMRMLVolumeNode.html
    imageData->SetSpacing(cellSize, cellSize, cellSize);
    imageData->SetOrigin(xmin + cellSize / 2.0f, ymin + cellSize / 2.0f, zmin + cellSize / 2.0f);

    //Apparently the vtkVolumeRayCastMapper class only works with unsigned char and unsigned short data
    imageData->AllocateScalars(VTK_UNSIGNED_INT, 1);

    // Fill every entry of the image data with a color
    start_ptr = static_cast<unsigned int*>(imageData->GetScalarPointer(0, 0, 0));

    double halfSize = cellSize / 2.0f;

    for(unsigned int x_index = 0; x_index < width; x_index++)
    {
        for(unsigned int y_index = 0; y_index < height; y_index++)
        {
            for(unsigned int z_index = 0; z_index < depth; z_index++)
            {
                //int x = std::floor(xmin + halfSize); x < std::ceil(xmax - halfSize); x = std::floor(x + cellSize)
                double pos_x = xmin + halfSize + x_index * cellSize;
                double pos_y = ymin + halfSize + y_index * cellSize;
                double pos_z = zmin + halfSize + z_index * cellSize;

                unsigned int* value_ptr = start_ptr + x_index + width * (y_index + z_index * height);

                *value_ptr = 0;

                data[x_index + width * (y_index + z_index * height)] = new Cell(
                        Point(pos_x, pos_y, pos_z),
                        cellSize,
                        value_ptr,
                        this,
                        &VisitationMap::cellModifiedCallback
                );
            }
        }
    }

    std::cout << "Complete." << std::endl;
}

void VisitationMap::cellModifiedCallback()
{
    imageData->Modified();
}

Cell* VisitationMap::GetCell(unsigned int index) const
{
    return data[index];
}

Cell* VisitationMap::FindCell(const Point& point) const
{
    if(point.X < xmin || point.X > xmax || point.Y < ymin || point.Y > ymax || point.Z < zmin || point.Z > zmax)
    {
        return nullptr;
    }

    unsigned int x_index = std::floor((point.X - xmin) / cellSize);
    unsigned int y_index = std::floor((point.Y - ymin) / cellSize);
    unsigned int z_index = std::floor((point.Z - zmin) / cellSize);

    unsigned int index = x_index + width * (y_index + z_index * height);
    return data[index];
}

unsigned int VisitationMap::GetNumberOfCells() const
{
    return width * height * depth;
}

vtkSmartPointer<vtkImageData> VisitationMap::GetImageData() const
{
    return imageData;
}