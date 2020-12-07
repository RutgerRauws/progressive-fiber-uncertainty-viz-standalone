//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_RENDERER_H

#include "glm/vec3.hpp"
#include "src/util/GL.h"
#include "../util/RenderElement.h"
#include "DWISlice.h"

class DWIRenderer : public RenderElement
{
    private:
        static constexpr auto   VERTEX_SHADER_PATH         = "./shaders/dwi/vertex.glsl";
        static constexpr auto   FRAGMENT_SHADER_PATH       = "./shaders/dwi/fragment.glsl";

        GL& gl;
        const DWISlice& slice;

        GLuint texture_loc = 0;
        GLuint opacity_loc;

        bool& showSlice;

        void initialize() override;

public:
        DWIRenderer(GL& gl, const Camera& camera, const DWISlice& slice, bool& showSlice);
        ~DWIRenderer();

        void Render() override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_DWI_RENDERER_H
