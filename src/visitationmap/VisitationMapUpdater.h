//
// Created by rutger on 10/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H

#include <string>
#include <GL/glew.h>

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

        const std::string COMPUTE_SHADER_PATH = "./shaders/compute.glsl";

        //DTI/DWI volume dimensions from example data set
//        const unsigned int width  = 112;
//        const unsigned int height = 112;
//        const unsigned int depth  = 70;
//        const float spacing = 2;

        double xmin, xmax, ymin, ymax, zmin, zmax;
        double spacing;

        unsigned int width;
        unsigned int height;
        unsigned int depth;

        void initialize();

        static std::string readStringFromFile(const std::string& path);
        static void checkForErrors(GLuint shader);

    public:
        VisitationMapUpdater(double* bounds, double spacing);
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
