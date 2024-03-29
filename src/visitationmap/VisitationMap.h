//
// Created by rutger on 10/15/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H

#include <src/util/GL.h>
#include "glm/vec3.hpp"
#include "AxisAlignedBoundingBox.h"

class VisitationMap
{
private:
    //!Changing these definitions also requires changing the definitions in the shader code!
    static const unsigned int NUMBER_OF_REPRESENTATIVE_FIBERS = 25;
    struct Cell
    {
        GLuint numberOfFibers = 0;                                      // 4 bytes
        GLuint representativeFibers[NUMBER_OF_REPRESENTATIVE_FIBERS];   // NUMBER_OF_REPRESENTATIVE_FIBERS * 4 bytes
    };

    GL& gl;

    GLint xmin, xmax, ymin, ymax, zmin, zmax; //coordinates in image space
    GLfloat spacing; //side length of a cell
    GLuint width, height, depth; //number of cells in each direction

    Cell* cell_data;
    GLuint cells_ssbo;

    unsigned int getCellIndex(unsigned int x_index, unsigned int y_index, unsigned int z_index) const;
    void getIndices(const glm::vec3& point, unsigned int& x_index, unsigned int& y_index, unsigned int& z_index) const;

    void initialize();
    void makeSphere();

public:
    /***
     * Set up the visitation map's memory layout on the GPU.
     * @param gl        Reference to the OpenGL functions
     * @param xmin      Minimum x-coordinate of the visitation map in world coordinates.
     * @param xmax      Maximum x-coordinate of the visitation map in world coordinates.
     * @param ymin      Minimum y-coordinate of the visitation map in world coordinates.
     * @param ymax      Maximum y-coordinate of the visitation map in world coordinates.
     * @param zmin      Minimum z-coordinate of the visitation map in world coordinates.
     * @param zmax      Maximum z-coordinate of the visitation map in world coordinates.
     * @param spacing   The side length of a single cell.
     */
    VisitationMap(GL& gl, GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax, GLfloat spacing);
    ~VisitationMap();

    static VisitationMap CreateVisitationMapFromDWIDimensions(GL& gl,
                                                              unsigned int x, unsigned int y, unsigned int z,
                                                              float dwi_spacing, float vm_spacing);

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

    Cell* GetData() const { return cell_data; }

    unsigned int GetNumberOfBytes() const { return sizeof(Cell) * GetWidth() * GetHeight() * GetDepth();}

    GLuint GetSSBOId() const { return cells_ssbo; }
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_H
