//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H

#include "glm/vec3.hpp"

#include "../util/RenderElement.h"
#include "../interaction/KeyPressObserver.h"
#include "../util/FiberObserver.h"
#include "../centerline/DistanceTableCollection.h"
#include "VisitationMap.h"

class VisitationMapRenderer : public RenderElement, public FiberObserver
{
    private:
        static constexpr auto   VERTEX_SHADER_PATH         = "./shaders/visitationmap/vertex.glsl";
        static constexpr auto   FRAGMENT_SHADER_PATH       = "./shaders/visitationmap/fragment.glsl";

        GL& gl;

        VisitationMap& visitationMap;
        RegionsOfInterest& regionsOfInterest;
        const DistanceTableCollection& distanceTables;

        GLint cameraPos_loc = -1;
        GLint frequency_isovalue_loc = -1;
        GLint distance_score_isovalue_loc = -1;
        GLint use_frequency_isovalue_loc = -1;
        GLint use_interpolcation_loc = -1;
        GLint opacity_loc = -1;
        GLint k_diffuse_loc = -1;
        GLint k_ambient_loc = -1;
        GLint k_specular_loc = -1;

        unsigned int numberOfFibers;

        void createVertices();
        void initialize() override;

        float computeFrequencyIsovalue() const;
        double computeDistanceScoreIsovalue() const;


public:
        VisitationMapRenderer(GL& gl,
                              VisitationMap& visitationMap,
                              RegionsOfInterest& regionsOfInterest,
                              const DistanceTableCollection& distanceTables,
                              const Camera& camera);
        ~VisitationMapRenderer();

        void Render() override;

        void NewFiber(Fiber* fiber) override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
