//
// Created by rutger on 11/9/20.
//

#include "DWIDataReader.h"

#include <string>
#include <iostream>

#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkMetaDataObject.h>

typedef itk::Vector<signed short, 2>             VectorType;
typedef itk::Image<VectorType, 3>	             DiffusionImageType;
typedef DiffusionImageType::Pointer	             DiffusionImagePointer;
typedef itk::ImageFileReader<DiffusionImageType> FileReaderType;

DWIDataReader::DWIDataReader(const std::string& fileName)
{
    FileReaderType::Pointer reader = FileReaderType::New();
    reader->SetFileName(fileName);
    reader->Update();

    auto imageSource = reader->GetOutput();
    std::cout << imageSource->GetNumberOfComponentsPerPixel() << std::endl;

    const DiffusionImageType::IndexType pixelIndex = {
            { 27, 29, 37 }
    };
    auto test = imageSource->GetPixel(pixelIndex);

    std::cout << test[0] << ", " << test[1] << std::endl;
}