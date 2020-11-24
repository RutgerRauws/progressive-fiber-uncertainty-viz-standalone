//
// Created by rutger on 11/24/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_SLICE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_SLICE_H

#include <GL/gl.h>

class DWISlice
{
private:
    GLubyte* data;

    unsigned int width, height;
    float widthWC, heightWC, depthWC;

public:
    DWISlice(unsigned int width, unsigned int height, float widthWC, float heightWC, float depthWC)
            : width(width), height(height),
              widthWC(widthWC), heightWC(heightWC), depthWC(depthWC)
    {
        data = new GLubyte[width * height * 4]; //RGBA
    };

    ~DWISlice()
    {
        delete[] data;
    };

    void SetPixel(unsigned int x, unsigned int y, GLubyte value)
    {
        data[(y * width + x) * 4]     = value;
        data[(y * width + x) * 4 + 1] = value;
        data[(y * width + x) * 4 + 2] = value;
        data[(y * width + x) * 4 + 3] = 255;
    };

    float* GetVertices() const
    {
        float halfWidth = widthWC / 2.0f;
        float halfDepth = depthWC / 2.0f;
        float halfHeight = heightWC / 2.0f;

        if(halfWidth == 0.0f)
        {
            return new float[5*6] {
                0, -halfDepth, -halfHeight, 0.0f, 0.0f,
                0,  halfDepth, -halfHeight, 1.0f, 0.0f,
                0,  halfDepth,  halfHeight, 1.0f, 1.0f,
                0,  halfDepth,  halfHeight, 1.0f, 1.0f,
                0, -halfDepth,  halfHeight, 0.0f, 1.0f,
                0, -halfDepth, -halfHeight, 0.0f, 0.0f
            };
        }
        else if(halfDepth == 0.0f)
        {
            return new float[5*6] {
                -halfWidth, 0, -halfHeight, 0.0f, 0.0f,
                 halfWidth, 0, -halfHeight, 1.0f, 0.0f,
                 halfWidth, 0,  halfHeight, 1.0f, 1.0f,
                 halfWidth, 0,  halfHeight, 1.0f, 1.0f,
                -halfWidth, 0,  halfHeight, 0.0f, 1.0f,
                -halfWidth, 0, -halfHeight, 0.0f, 0.0f
            };
        }
        else if(halfHeight == 0.0f)
        {
            return new float[5*6] {
                -halfWidth, -halfDepth, 0, 0.0f, 0.0f,
                 halfWidth, -halfDepth, 0, 1.0f, 0.0f,
                 halfWidth,  halfDepth, 0, 1.0f, 1.0f,
                 halfWidth,  halfDepth, 0, 1.0f, 1.0f,
                -halfWidth,  halfDepth, 0, 0.0f, 1.0f,
                -halfWidth, -halfDepth, 0, 0.0f, 0.0f
            };
        }
        else
        {
            throw std::runtime_error("DWI slices do not pass through the origin.");
        }
    }

    GLubyte* GetBytes() const { return data; };
    unsigned int GetWidth() const { return width; };
    unsigned int GetHeight() const { return height; };
};



#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_SLICE_H
