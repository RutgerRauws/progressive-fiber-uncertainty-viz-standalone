//
// Created by rutger on 10/15/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H

#include "../util/glm/vec3.hpp"
#include "AxisAlignedBoundingBox.h"
#include "CellFiberMultiMap.h"

class VisitationMap
{
private:
    GLint xmin, xmax, ymin, ymax, zmin, zmax;
    GLfloat spacing;
    GLuint width, height, depth;

    GLuint* frequency_data;

    GLuint frequency_map_ssbo;

    CellFiberMultiMap cellFiberMultiMap;

    unsigned int getCellIndex(unsigned int x_index, unsigned int y_index, unsigned int z_index) const;
    void getIndices(const glm::vec3& point, unsigned int& x_index, unsigned int& y_index, unsigned int& z_index) const;

    void initialize();
    void makeSphere();

public:
    VisitationMap(GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax, GLfloat spacing);
    ~VisitationMap();

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

    unsigned int GetNumberOfBytes() const { return sizeof(unsigned int) * GetWidth() * GetHeight() * GetDepth();}

    GLuint GetFrequencyMapSSBOId() const { return frequency_map_ssbo; }

    CellFiberMultiMap& GetCellFiberMultiMap() { return cellFiberMultiMap; }
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
