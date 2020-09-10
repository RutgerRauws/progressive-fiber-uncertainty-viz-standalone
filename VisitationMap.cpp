//
// Created by rutger on 7/20/20.
//

#include "VisitationMap.h"
#include "Point.h"
#include "Cell.h"

#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkInformationVector.h>
#include <vtkInformation.h>
#include <vtkDataObject.h>
#include <vtkSmartPointer.h>

VisitationMap::VisitationMap(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
    : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax),
      imageData(vtkSmartPointer<vtkImageData>::New())
{
    //TODO: Look into fixing double to int conversion.
    width =  std::ceil(std::abs(xmin - xmax) / cellSize);
    height = std::ceil(std::abs(ymin - ymax) / cellSize);
    depth =  std::ceil(std::abs(zmin - zmax) / cellSize);

    this->SetNumberOfInputPorts(0);
    this->SetNumberOfOutputPorts(1);

    initialize();
}

/**
 *
 * @param bounds xmin,xmax, ymin,ymax, zmin,zmax
 */
VisitationMap::VisitationMap(double* bounds)
    : VisitationMap(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
{}

//int VisitationMap::RequestInformation(vtkInformation* vtkNotUsed(request),
//  vtkInformationVector** vtkNotUsed(inputVector), vtkInformationVector* outputVector)
//{
//    vtkInformation* outInfo = outputVector->GetInformationObject(0);
//    outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->imageData->GetExtent(), 6);
//
//    return 1;
//}
//

int VisitationMap::RequestData(vtkInformation *vtkNotUsed(request),
                                         vtkInformationVector **vtkNotUsed(inputVector),
                                         vtkInformationVector *outputVector)
{
    // Get the info objects
//    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
    vtkInformation *outInfo = outputVector->GetInformationObject(0);

    // Get the input and ouptut
//    vtkImageData *input = dynamic_cast<vtkImageData*>(
//            inInfo->Get(vtkDataObject::DATA_OBJECT()));

//    vtkImageData *output = dynamic_cast<vtkImageData*>(
//            outInfo->Get(vtkDataObject::DATA_OBJECT()));

    vtkImageData *output = vtkImageData::SafeDownCast(
            outInfo->Get(vtkDataObject::DATA_OBJECT())
    );


//    vtkSmartPointer<vtkImageData> image =
//            vtkSmartPointer<vtkImageData>::New();
//    image->ShallowCopy(input);
//
//    image->SetScalarComponentFromDouble(0,0,0,0, 5.0);
//
//    output->ShallowCopy(image);

    output->ShallowCopy(imageData);
    //output->DeepCopy(imageData);
    // Without these lines, the output will appear real but will not work as the input to any other filters
//                                    int extent[6];
//                                //    input->GetExtent(extent);
//                                    imageData->GetExtent(extent);
//                                    output->SetExtent(extent);
//                                    outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
//                                                 extent,
//                                                 6);
//                                    outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
//                                                 extent,
//                                                 6);

    // Without these lines, the output will appear real but will not work as the input to any other filters
    output->SetExtent(imageData->GetExtent());
//    output->SetUpdateExtent(output->GetExtent());
//    output->SetWholeExtent(output->GetExtent());

    return 1;
}

void VisitationMap::initialize()
{
    std::cout << "Initializing visitation map... " << std::flush;

    data = new Cell*[GetNumberOfCells()];

    imageData->SetExtent(0, width - 1, 0, height - 1, 0, depth - 1);
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
    this->Modified();
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

    for(unsigned int index = 0; index < GetNumberOfCells(); index++)
    {
        if(data[index]->Contains(point))
        {
            return data[index];
        }
    }

//    //TODO: Make use of acceleration data structures such as octrees.
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

vtkSmartPointer<vtkImageData> VisitationMap::GetImageData() const
{
    return imageData;
}