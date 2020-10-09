//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_BOX_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_BOX_H

#include "RenderElement.h"

class Box : public RenderElement
{
    private:
        float xmin, xmax, ymin, ymax, zmin, zmax;

        void createVertices();
        void initialize() override;

    public:
        Box(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);
        explicit Box(float* bounds);
        ~Box();

        void Render() override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_BOX_H
