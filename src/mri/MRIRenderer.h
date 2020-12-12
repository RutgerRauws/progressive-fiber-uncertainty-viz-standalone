//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_RENDERER_H

#include "glm/vec3.hpp"
#include "src/util/GL.h"
#include "../util/RenderElement.h"
#include "MRISlice.h"

class MRIRenderer : public RenderElement
{
    private:
        static constexpr auto   VERTEX_SHADER_PATH         = "./shaders/mri/vertex.glsl";
        static constexpr auto   FRAGMENT_SHADER_PATH       = "./shaders/mri/fragment.glsl";

        GL& gl;
        const MRISlice& slice;

        GLuint texture_loc = 0;
        GLuint opacity_loc;

        bool& showSlice;

        void initialize() override;

public:
        MRIRenderer(GL& gl, const Camera& camera, const MRISlice& slice, bool& showSlice);
        ~MRIRenderer();

        void Render() override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_RENDERER_H
