//
// Created by rutger on 7/20/20.
//

#include "VisitationMap.h"
#include "Point.h"
#include "Cell.h"

VisitationMap::VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
    : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax),
      fiberFrequencyImageData(vtkSmartPointer<vtkImageData>::New()),
      distanceScoreImageData(vtkSmartPointer<vtkImageData>::New())
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

VisitationMap::~VisitationMap()
{
    for(unsigned int i = 0; i < GetNumberOfCells(); i++)
    {
        delete GetCell(i);
    }

    delete[] cells;
}

void VisitationMap::initialize()
{
    std::cout << "Initializing visitation map... " << std::flush;

    cells = new Cell*[GetNumberOfCells()];

    fiberFrequencyImageData->SetExtent(0, width - 1, 0, height - 1, 0, depth - 1);
    fiberFrequencyImageData->SetSpacing(cellSize, cellSize, cellSize);
    fiberFrequencyImageData->SetOrigin(xmin + cellSize / 2.0f, ymin + cellSize / 2.0f, zmin + cellSize / 2.0f);

    distanceScoreImageData->SetExtent(0, width - 1, 0, height - 1, 0, depth - 1);
    distanceScoreImageData->SetSpacing(cellSize, cellSize, cellSize);
    distanceScoreImageData->SetOrigin(xmin + cellSize / 2.0f, ymin + cellSize / 2.0f, zmin + cellSize / 2.0f);

    //Apparently the vtkVolumeRayCastMapper class only works with unsigned char and unsigned short cells
    fiberFrequencyImageData->AllocateScalars(VTK_UNSIGNED_INT, 1);
    distanceScoreImageData->AllocateScalars(VTK_DOUBLE, 1);

    // Fill every entry of the image cells with a color
    fiberFrequencyStart_ptr = static_cast<unsigned int*>(fiberFrequencyImageData->GetScalarPointer(0, 0, 0));
    distanceScoreStart_ptr = static_cast<double*>(distanceScoreImageData->GetScalarPointer(0, 0, 0));

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

                unsigned int* fiberFrequency_ptr = fiberFrequencyStart_ptr + x_index + width * (y_index + z_index * height);
                double* distanceScore_ptr = distanceScoreStart_ptr + x_index + width * (y_index + z_index * height);

                *fiberFrequency_ptr = 1;
                *distanceScore_ptr = std::numeric_limits<unsigned int>::max();

                cells[x_index + width * (y_index + z_index * height)] = new Cell(
                        Point(pos_x, pos_y, pos_z),
                        cellSize,
                        fiberFrequency_ptr,
                        distanceScore_ptr,
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
    fiberFrequencyImageData->Modified();
    distanceScoreImageData->Modified();
}

Cell* VisitationMap::GetCell(unsigned int index) const
{
    return cells[index];
}

Cell* VisitationMap::FindCell(const Point& point) const
{
    if(point.X < xmin || point.X > xmax || point.Y < ymin || point.Y > ymax || point.Z < zmin || point.Z > zmax)
    {
        return nullptr;
    }

    for(unsigned int index = 0; index < GetNumberOfCells(); index++)
    {
        if(cells[index]->Contains(point))
        {
            return cells[index];
        }
    }

//    //TODO: Make use of acceleration cells structures such as octrees.
//    for(unsigned int z_index = 0; z_index < depth + 1; z_index++)
//    {
//        for(unsigned int y_index = 0; y_index < height + 1; y_index++)
//        {
//            for(unsigned int x_index = 0; x_index < width + 1; x_index++)
//            {
//
//
//                if(containedInCell(x_index, y_index, z_index, point))
//                {
//                    return getCell(x_index, y_index, z_index);
//                }
//            }
//        }
//    }

    return nullptr;
}

unsigned int VisitationMap::GetNumberOfCells() const
{
    return width * height * depth;
}

vtkSmartPointer<vtkImageData> VisitationMap::GetFiberFrequencyImageData() const
{
    return fiberFrequencyImageData;
}

vtkSmartPointer<vtkImageData> VisitationMap::GetDistanceScoreImageData() const
{
    return distanceScoreImageData;
}