//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H

#include "../util/RenderElement.h"
#include "../util/glm/vec3.hpp"
#include "../interaction/KeyPressObserver.h"
#include "VisitationMap.h"

class VisitationMapRenderer : public RenderElement
{
    private:
        #define VERTEX_SHADER_PATH   "./shaders/visitationmap/vertex.glsl"
        #define FRAGMENT_SHADER_PATH "./shaders/visitationmap/fragment.glsl"

        VisitationMap& visitationMap;

        GLint cameraPos_loc = -1;

        void createVertices();
        void initialize() override;

    public:
        VisitationMapRenderer(VisitationMap& visitationMap, const CameraState& cameraState);
        ~VisitationMapRenderer();

        void Render() override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
