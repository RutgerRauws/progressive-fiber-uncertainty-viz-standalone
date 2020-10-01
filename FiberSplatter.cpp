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
#include <functional>

vtkStandardNewMacro(FiberSplatter);

FiberSplatter::FiberSplatter()
    : CellSize(1), KernelRadius(3)
{}

void FiberSplatter::SetExistingPointsFiberData(std::vector<std::vector<std::reference_wrapper<const Fiber>>>& fibers)
{
    this->ExistingPointsFibers = &fibers;
}

void FiberSplatter::SetCellFrequencies(vtkUnsignedIntArray* cellFrequencies)
{
    this->CellFrequencies = cellFrequencies;
}

int FiberSplatter::RequestData(vtkInformation* vtkNotUsed(request),
                               vtkInformationVector** inputVector,
                               vtkInformationVector* outputVector)
{
    // get the data object
    vtkInformation* outInfo = outputVector->GetInformationObject(0);
    vtkImageData* output = vtkImageData::GetData(outputVector, 0);

    vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
    vtkDataSet* inputDS = vtkDataSet::GetData(inInfo);
//    vtkPolyData* inputPolyData = vtkPolyData::GetData(inInfo);
    vtkNew<vtkPolyData> inputPolyData;
    inputPolyData->DeepCopy(vtkPolyData::GetData(inInfo));

    int extent[6];
    getExtent(inputPolyData->GetBounds(), extent);
    output->SetExtent(extent);

    output->SetExtent(extent);

    // TODO: Perhaps the SetSpacing and SetOrigin calls should be removed.
    // As we use vtkMRMLVolumeNode::SetAndObserveImageData, the ImageData object origin must be set to (0,0,0) and
    // spacing must be set to (1,1,1). If the variables are set to different values then the application's behavior is
    // undefined.
    // https://apidocs.slicer.org/v4.8/classvtkMRMLVolumeNode.html
    output->SetSpacing(CellSize, CellSize, CellSize);
    output->SetOrigin(CellSize / 2.0f, CellSize / 2.0f, CellSize / 2.0f);

    //Apparently the vtkVolumeRayCastMapper class only works with unsigned char and unsigned short data
    output->AllocateScalars(VTK_UNSIGNED_INT, 1);

    auto scalars_ptr = static_cast<unsigned int*>(output->GetScalarPointer());
    std::fill_n(scalars_ptr, output->GetNumberOfPoints(), 0);

    vtkIdType inNumPts = inputPolyData->GetNumberOfPoints();
    if (inNumPts == 0)
    {
        return 1;
    }

    std::vector<std::vector<Fiber*>> outPointFibers(output->GetNumberOfPoints());

    for(vtkIdType inPtId = 0; inPtId < inNumPts; inPtId++)
    {
        double* inPt = inputPolyData->GetPoint(inPtId);

        int index_x = std::floor(inPt[0] / CellSize);
        int index_y = std::floor(inPt[1] / CellSize);
        int index_z = std::floor(inPt[2] / CellSize);

        auto value_ptr = static_cast<unsigned int*>(output->GetScalarPointer(index_x, index_y, index_z));

//        auto& test = ExistingPointsFibers[inPtId];
//        *value_ptr = test.size();
        //*value_ptr = CellFrequencies->GetValue(inPtId);
        *value_ptr = 2;

//        std::cout << *value_ptr << std::endl;
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
    extent[0] = std::floor((bounds[0] - CellSize) / CellSize); //xmin
    extent[1] = std::ceil( (bounds[1] + CellSize) / CellSize); //xmax
    extent[2] = std::floor((bounds[2] - CellSize) / CellSize); //ymin
    extent[3] = std::ceil( (bounds[3] + CellSize) / CellSize); //ymax
    extent[4] = std::floor((bounds[4] - CellSize) / CellSize); //zmin
    extent[5] = std::ceil( (bounds[5] + CellSize) / CellSize); //zmax
}