//
// Created by rutger on 11/9/20.
//

#include "MRIDataReader.h"
#include "MRISlice.h"

MRIDataReader::MRIDataReader(const std::string& fileName, float normalization_denominator)
    : normalization_denominator(normalization_denominator)
{
    FileReaderType::Pointer reader = FileReaderType::New();
    reader->SetFileName(fileName);
    reader->Update();

    imageSource = reader->GetOutput();
    auto dimensions = imageSource->GetLargestPossibleRegion().GetSize();

    size_x = dimensions[0];
    size_y = dimensions[2];
    size_z = dimensions[1];

    spacing = imageSource->GetSpacing()[0];

    size_x_wc  = spacing * (float)size_x;
    size_y_wc = spacing * (float)size_y;
    size_z_wc  = spacing * (float)size_z;
}

MRISlice MRIDataReader::GetCoronalPlane() const
{
    MRISlice slice(size_x, size_y, size_x_wc, size_y_wc, 0);

    int y = size_z / 2.0;

    for (int x = 0; x < size_x; x++)
    {
        for (int z = 0; z < size_y; z++)
        {
            const DiffusionImageType::IndexType pixelIndex = {
                    {x, y, z}
            };

            auto pixel = imageSource->GetPixel(pixelIndex);

            float value = pixel / normalization_denominator;

            slice.SetPixel(x, z, value);
        }
    }

    return slice;
}

MRISlice MRIDataReader::GetAxialPlane() const
{
    MRISlice slice(size_x, size_z, size_x_wc, 0, size_z_wc);

    int z = size_y / 2.0f;

    for (int x = 0; x < size_x; x++)
    {
        for (int y = 0; y < size_z; y++)
        {
            const DiffusionImageType::IndexType pixelIndex = {
                    {x, y, z}
            };

            auto pixel = imageSource->GetPixel(pixelIndex);

            float value = pixel / normalization_denominator;

            slice.SetPixel(x, size_z - y - 1, value);
        }
    }

    return slice;
}

MRISlice MRIDataReader::GetSagittalPlane() const
{
    MRISlice slice(size_z, size_y, 0, size_y_wc, size_z_wc);

    int x = size_x / 2.0;

    for (int y = 0; y < size_z; y++)
    {
        for (int z = 0; z < size_y; z++)
        {
            const DiffusionImageType::IndexType pixelIndex = {
                    {x, y, z}
            };

            auto pixel = imageSource->GetPixel(pixelIndex);

            float value = pixel / normalization_denominator;

            slice.SetPixel(size_z - y - 1, z, value);
        }
    }

    return slice;
}
