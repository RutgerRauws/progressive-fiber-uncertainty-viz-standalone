//
// Created by rutger on 10/25/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_AXIS_ALIGNED_BOUNDING_BOX_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_AXIS_ALIGNED_BOUNDING_BOX_H

#include <GL/glew.h>
#include <limits>
#include <vector>

class AxisAlignedBoundingBox
{
    public:
        GLint xmin, xmax, ymin, ymax, zmin, zmax;

        AxisAlignedBoundingBox() = default;

        AxisAlignedBoundingBox(GLint xmin, GLint xmax, GLint ymin, GLint ymax, GLint zmin, GLint zmax)
            : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax)
        {}

        static AxisAlignedBoundingBox CreateStartAABB()
        {
            AxisAlignedBoundingBox aabb(
                std::numeric_limits<GLint>::max(),
                std::numeric_limits<GLint>::min(),
                std::numeric_limits<GLint>::max(),
                std::numeric_limits<GLint>::min(),
                std::numeric_limits<GLint>::max(),
                std::numeric_limits<GLint>::min()
            );

            return aabb;
        }
};

class RegionsOfInterest
{
    private:
        GLuint rois_ssbo_id;

        unsigned int numberOfEnsembles;
        std::vector<AxisAlignedBoundingBox> roiAABBs;

    public:
        RegionsOfInterest(unsigned int numberOfEnsembles) : numberOfEnsembles(numberOfEnsembles)
        {
            glGenBuffers(1, &rois_ssbo_id);
            for(unsigned int i = 0; i < numberOfEnsembles; i++) { roiAABBs.emplace_back(AxisAlignedBoundingBox::CreateStartAABB()); }
        }

        void Initialize()
        {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, rois_ssbo_id);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(AxisAlignedBoundingBox) * numberOfEnsembles, roiAABBs.data(), GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rois_ssbo_id);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind
        }

        GLuint GetSSBOId() const { return rois_ssbo_id; }
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_AXIS_ALIGNED_BOUNDING_BOX_H
