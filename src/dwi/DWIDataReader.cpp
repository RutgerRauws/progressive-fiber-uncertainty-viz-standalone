//
// Created by rutger on 11/9/20.
//

#include "DWIDataReader.h"
#include "DWISlice.h"

DWIDataReader::DWIDataReader(const std::string& fileName)
{
    FileReaderType::Pointer reader = FileReaderType::New();
    reader->SetFileName(fileName);
    reader->Update();

    imageSource = reader->GetOutput();
    auto dimensions = imageSource->GetLargestPossibleRegion().GetSize();

    width = dimensions[0];
    height = dimensions[2];
    depth = dimensions[1];

    spacing = imageSource->GetSpacing()[0];

    widthWC  = spacing * (float)width;
    heightWC = spacing * (float)height;
    depthWC  = spacing * (float)depth;
}

DWISlice DWIDataReader::GetCoronalPlane() const
{
    DWISlice slice(width, height, widthWC, heightWC, 0);

    int y = depth / 2.0;

    for (int x = 0; x < width; x++)
    {
        for (int z = 0; z < height; z++)
        {
            const DiffusionImageType::IndexType pixelIndex = {
                    {x, y, z}
            };

            auto pixel = imageSource->GetPixel(pixelIndex);

            float value = pixel / 20.0f;

            slice.SetPixel(x, z, value);
        }
    }

    return slice;
}

DWISlice DWIDataReader::GetAxialPlane() const
{
    DWISlice slice(width, depth, widthWC, 0, depthWC);

    int z = height / 2.0f;

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < depth; y++)
        {
            const DiffusionImageType::IndexType pixelIndex = {
                    {x, y, z}
            };

            auto pixel = imageSource->GetPixel(pixelIndex);

            float value = pixel / 20.0f;

            slice.SetPixel(x, y, value);
        }
    }

    return slice;
}

DWISlice DWIDataReader::GetSagittalPlane() const
{
    DWISlice slice(depth, height, 0, heightWC, depthWC);

    int x = width / 2.0;

    for (int y = 0; y < depth; y++)
    {
        for (int z = 0; z < height; z++)
        {
            const DiffusionImageType::IndexType pixelIndex = {
                    {x, y, z}
            };

            auto pixel = imageSource->GetPixel(pixelIndex);

            float value = pixel / 20.0f;

            slice.SetPixel(y, z, value);
        }
    }

    return slice;
}
