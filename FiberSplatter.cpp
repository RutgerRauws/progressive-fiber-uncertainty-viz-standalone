//
// Created by rutger on 9/27/20.
//

#include "FiberSplatter.h"

#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkImageData.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkPolyData.h>
#include <vtkObjectFactory.h>

vtkStandardNewMacro(FiberSplatter);

FiberSplatter::FiberSplatter()
    : CellSize(1), KernelRadius(3)
{}

void FiberSplatter::SetExistingPointsFiberData(std::vector<std::vector<std::reference_wrapper<const Fiber>>>& fibers)
{
    this->ExistingPointsFibers = &fibers;
}

int FiberSplatter::RequestData(vtkInformation* vtkNotUsed(request),
                               vtkInformationVector** inputVector,
                               vtkInformationVector* outputVector)
{
    // get the data object
    vtkInformation* outInfo = outputVector->GetInformationObject(0);
    vtkImageData* output = vtkImageData::GetData(outputVector, 0);

    output->SetExtent(outInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()));
    output->AllocateScalars(outInfo);

    vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
    vtkDataSet* inputDS = vtkDataSet::GetData(inInfo);
    vtkPolyData* inputPolyData = vtkPolyData::GetData(inInfo);

    vtkIdType inNumPts = inputPolyData->GetNumberOfPoints();

    if (inNumPts == 0)
    {
        return 1;
    }

//    output->SetDimensions(this->GetSampleDimensions());
    int extent[6];
    getExtent(inputPolyData->GetBounds(), extent);
    output->SetExtent(extent);

    int xmin = extent[0];
    int xmax = extent[1];
    int ymin = extent[2];
    int ymax = extent[3];
    int zmin = extent[4];
    int zmax = extent[5];

    output->SetSpacing(CellSize, CellSize, CellSize);
    output->AllocateScalars(VTK_UNSIGNED_INT, 1);

    vtkIdType outNumPts = std::abs(xmin) * std::abs(xmax) * std::abs(ymin) * std::abs(ymax) * std::abs(zmin) * std::abs(zmax);

    std::vector<std::vector<Fiber*>> outPointFibers(outNumPts);

//    auto start_ptr = static_cast<unsigned int*>(output->GetScalarPointer(0, 0, 0));

    for(vtkIdType inPtId = 0; inPtId < inNumPts; inPtId++)
    {
        //output->GetPoint()
        double* inPt = inputPolyData->GetPoint(inPtId);

        vtkIdType outPtId = output->FindPoint(inPt);
        double* outPt = output->GetPoint(outPtId);

//        int index_x = std::floor((outPt[0] - xmin) / CellSize);
//        int index_y = std::floor((outPt[1] - ymin) / CellSize);
//        int index_z = std::floor((outPt[2] - zmin) / CellSize);
        int index_x = std::floor((outPt[0]) / CellSize);
        int index_y = std::floor((outPt[1]) / CellSize);
        int index_z = std::floor((outPt[2]) / CellSize);

        //unsigned int* value_ptr = start_ptr + x_index + width * (y_index + z_index * height);
        auto value_ptr = static_cast<unsigned int*>(output->GetScalarPointer(index_x, index_y, index_z));

        *value_ptr = 1;
    }

    return 1;
}

int FiberSplatter::FillInputPortInformation(int vtkNotUsed(port), vtkInformation* info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
    info->Append(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkCompositeDataSet");
    return 1;
}

/***
 *
 * @param bounds  (xmin,xmax, ymin,ymax, zmin,zmax).
 * @param extent  (xmin,xmax, ymin,ymax, zmin,zmax).
 */
void FiberSplatter::getExtent(double* bounds, int* extent) const
{
    extent[0] = std::floor((bounds[0] - KernelRadius) / CellSize); //xmin
    extent[1] = std::ceil( (bounds[1] + KernelRadius) / CellSize); //xmax
    extent[2] = std::floor((bounds[2] - KernelRadius) / CellSize); //ymin
    extent[3] = std::ceil( (bounds[3] + KernelRadius) / CellSize); //ymax
    extent[4] = std::floor((bounds[4] - KernelRadius) / CellSize); //zmin
    extent[5] = std::ceil( (bounds[5] + KernelRadius) / CellSize); //zmax
}