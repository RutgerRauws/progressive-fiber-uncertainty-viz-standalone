//
// Created by rutger on 11/23/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_DATA_READER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_DATA_READER_H


#include <string>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include "MRISlice.h"

class MRIDataReader
{
    private:
        typedef short                                    VectorType;
        typedef itk::Image<VectorType, 3>	             DiffusionImageType;
        typedef DiffusionImageType::Pointer	             DiffusionImagePointer;
        typedef itk::ImageFileReader<DiffusionImageType> FileReaderType;

        DiffusionImagePointer imageSource;

        int size_x, size_y, size_z;
        float size_x_wc, size_y_wc, size_z_wc;

        float spacing;


    public:
        MRIDataReader(const std::string& fileName);

        MRISlice GetCoronalPlane() const;//or frontal plane
        MRISlice GetAxialPlane() const; //or horizontal plane
        MRISlice GetSagittalPlane() const;

        void CreatePNGs() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_DATA_READER_H
