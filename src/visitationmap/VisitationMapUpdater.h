//
// Created by rutger on 10/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H

#include <string>
#include <GL/glew.h>
#include "VisitationMap.h"

class VisitationMapUpdater
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

        VisitationMap& visitationMap;

        void initialize();

        static std::string readStringFromFile(const std::string& path);
        static void checkForErrors(GLuint shader);

    public:
        VisitationMapUpdater(VisitationMap& visitationMap);
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
