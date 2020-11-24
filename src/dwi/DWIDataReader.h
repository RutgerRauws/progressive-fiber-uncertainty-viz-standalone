//
// Created by rutger on 11/23/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_DATA_READER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_DATA_READER_H


#include <string>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include "DWISlice.h"

class DWIDataReader
{
    private:
        typedef short                                    VectorType;
        typedef itk::Image<VectorType, 3>	             DiffusionImageType;
        typedef DiffusionImageType::Pointer	             DiffusionImagePointer;
        typedef itk::ImageFileReader<DiffusionImageType> FileReaderType;

        DiffusionImagePointer imageSource;

        int width, height, depth;
        float widthWC, heightWC, depthWC;

        float spacing;


    public:
        DWIDataReader(const std::string& fileName);

        DWISlice GetCoronalPlane() const;//or frontal plane
        DWISlice GetAxialPlane() const; //or horizontal plane
        DWISlice GetSagittalPlane() const;

        void CreatePNGs() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_DATA_READER_H
