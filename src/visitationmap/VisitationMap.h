//
// Created by rutger on 10/15/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H

#include "../util/glm/vec3.hpp"

class VisitationMap
{
private:
    struct AxisAlignedBoundingBox
    {
        GLint xmin = std::numeric_limits<GLint>::max();
        GLint xmax = std::numeric_limits<GLint>::min();
        GLint ymin = std::numeric_limits<GLint>::max();
        GLint ymax = std::numeric_limits<GLint>::min();
        GLint zmin = std::numeric_limits<GLint>::max();
        GLint zmax = std::numeric_limits<GLint>::min();
    };

    //properties
    GLint xmin, xmax, ymin, ymax, zmin, zmax;
    GLfloat spacing;
    GLuint width, height, depth;

    GLuint* frequency_data;
    const AxisAlignedBoundingBox aabb_structure;

    GLuint frequency_map_ssbo;


    unsigned int getCellIndex(unsigned int x_index, unsigned int y_index, unsigned int z_index) const;
    void getIndices(const glm::vec3& point, unsigned int& x_index, unsigned int& y_index, unsigned int& z_index) const;

    void initialize();
    void makeSphere();

public:
    VisitationMap(GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax, GLfloat spacing);

    static VisitationMap CreateTest();
    static VisitationMap CreateVisitationMap(const glm::vec3& seedPoint, float cutoffLength);

    GLint GetXmin() const { return xmin; }
    GLint GetYmin() const { return ymin; }
    GLint GetZmin() const { return zmin; }
    GLint GetXmax() const { return xmax; }
    GLint GetYmax() const { return ymax; }
    GLint GetZmax() const { return zmax; }
    GLfloat GetSpacing() const { return spacing; }

    GLuint GetWidth() const { return width; }
    GLuint GetHeight() const { return height; }
    GLuint GetDepth() const { return depth; }

    GLuint* GetData() const { return frequency_data; }

    const AxisAlignedBoundingBox& GetAABB() const { return aabb_structure; }
    unsigned int GetNumberOfBytes() const { return sizeof(unsigned int) * GetWidth() * GetHeight() * GetDepth();}

    GLuint GetSSBOId() const { return frequency_map_ssbo; }
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
