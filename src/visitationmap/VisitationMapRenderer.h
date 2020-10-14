//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H

#include "../util/RenderElement.h"
#include "../util/glm/vec3.hpp"
#include "../interaction/KeyPressObserver.h"

class VisitationMapRenderer : public RenderElement
{
    private:
        float xmin, xmax, ymin, ymax, zmin, zmax;
        unsigned int width, height, depth;
        float spacing = 2.0;

        unsigned int* frequency_data;

        void createVertices();
        void initialize() override;

        unsigned int getCellIndex(unsigned int x_index, unsigned int y_index, unsigned int z_index);
        void getIndices(const glm::vec3& point, unsigned int& x_index, unsigned int& y_index, unsigned int& z_index);
        void makeSphere();

    public:
        VisitationMapRenderer(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);
        explicit VisitationMapRenderer(float* bounds);
        ~VisitationMapRenderer();

        void Render() override;

        void SetUpUniforms(GLuint programId);

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
