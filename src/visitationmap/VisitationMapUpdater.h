//
// Created by rutger on 10/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H

#include <string>
#include <GL/glew.h>
#include "VisitationMap.h"
#include "../util/Shader.h"
#include "../util/ShaderProgram.h"
#include "../util/FiberObserver.h"

class VisitationMapUpdater : public FiberObserver
{
    private:
//        struct VisitationMapProperties
//        {
//            VisitationMapProperties(int width, int height, int depth, float cellSize)
//                : width(width), height(height), depth(depth), cellSize(cellSize)
//            {}
//
//            int width, height, depth;
//            float cellSize;
//        };

        const std::string COMPUTE_SHADER_PATH = "./shaders/visitationmap/compute.glsl";

        Shader* computeShader = nullptr;
        ShaderProgram* shaderProgram = nullptr;

        VisitationMap& visitationMap;

        std::vector<Fiber*> fiberQueue;

        void initialize();

    public:
        explicit VisitationMapUpdater(VisitationMap& visitationMap);

        void NewFiber(Fiber* fiber) override;
        void Update();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
