//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H

#include "../util/RenderElement.h"
#include "../util/glm/vec3.hpp"
#include "../interaction/KeyPressObserver.h"
#include "VisitationMap.h"
#include "../util/FiberObserver.h"

class VisitationMapRenderer : public RenderElement, public KeyPressObserver, public FiberObserver
{
    private:
        static constexpr auto VERTEX_SHADER_PATH   = "./shaders/visitationmap/vertex.glsl";
        static constexpr auto FRAGMENT_SHADER_PATH = "./shaders/visitationmap/fragment.glsl";
        static constexpr float PERCENTAGE_DELTA = 0.01f;

        VisitationMap& visitationMap;
        RegionsOfInterest& regionsOfInterest;

        GLint cameraPos_loc = -1;
        GLint isovalue_loc = -1;

        float isovaluePercentage;
        unsigned int numberOfFibers;

        void createVertices();
        void initialize() override;
        void updateIsovaluePercentage(float delta);
        unsigned int computeIsovalue();

    public:
        VisitationMapRenderer(VisitationMap& visitationMap, RegionsOfInterest& regionsOfInterest, const CameraState& cameraState);
        ~VisitationMapRenderer();

        void Render() override;

        void KeyPressed(const sf::Keyboard::Key& key) override;
        void NewFiber(Fiber* fiber) override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_RENDERER_H
