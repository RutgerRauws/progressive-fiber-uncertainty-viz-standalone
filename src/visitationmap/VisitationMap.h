//
// Created by rutger on 10/15/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H

#include "../util/glm/vec3.hpp"

class VisitationMap
{
private:
    float xmin, xmax, ymin, ymax, zmin, zmax;
    float spacing;
    unsigned int width, height, depth;

    unsigned int* frequency_data;

    GLuint frequency_map_ssbo;


    unsigned int getCellIndex(unsigned int x_index, unsigned int y_index, unsigned int z_index) const;
    void getIndices(const glm::vec3& point, unsigned int& x_index, unsigned int& y_index, unsigned int& z_index) const;

    void initialize();
    void makeSphere();

public:
    VisitationMap(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float spacing);

    static VisitationMap CreateTest();

    float GetXmin() const { return xmin; }
    float GetYmin() const { return ymin; }
    float GetZmin() const { return zmin; }
    float GetXmax() const { return xmax; }
    float GetYmax() const { return ymax; }
    float GetZmax() const { return zmax; }
    float GetSpacing() const {return spacing; }

    unsigned int GetWidth() const { return width; }
    unsigned int GetHeight() const { return height; }
    unsigned int GetDepth() const { return depth; }

    unsigned int* GetData() const { return frequency_data; }

    GLuint GetSSBOId() const { return frequency_map_ssbo; }
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
