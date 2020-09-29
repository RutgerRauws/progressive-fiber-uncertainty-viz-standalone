//
// Created by rutger on 9/27/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_SPLATTER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_SPLATTER_H


#include <vtkImageAlgorithm.h>
#include <vector>
#include <vtkUnsignedIntArray.h>
#include "Fiber.h"

class FiberSplatter : public vtkImageAlgorithm
{
    public:
        double CellSize;
        double KernelRadius;

        vtkTypeMacro(FiberSplatter, vtkImageAlgorithm);

        static FiberSplatter* New();

        void SetExistingPointsFiberData(std::vector<std::vector<std::reference_wrapper<const Fiber>>>& fibers);
        void SetCellFrequencies(vtkUnsignedIntArray* cellFrequencies);

        //int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
        int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
        int FillInputPortInformation(int port, vtkInformation* info) override;

    private:
        std::vector<std::vector<std::reference_wrapper<const Fiber>>>* ExistingPointsFibers;
        vtkUnsignedIntArray* CellFrequencies;

        void getExtent(double* bounds, int* extent) const;

    private:
        FiberSplatter(const FiberSplatter&) = delete;
        void operator=(const FiberSplatter&) = delete;

    protected:
        FiberSplatter();
        ~FiberSplatter() override = default;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_SPLATTER_H
