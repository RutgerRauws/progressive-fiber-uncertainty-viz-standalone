//
// Created by rutger on 10/15/20.
//

#include "VisitationMap.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>
#include <Configuration.h>
#include "glm/glm.hpp"

VisitationMap::VisitationMap(GL& gl, GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax, GLfloat spacing)
    : gl(gl),
      xmin(std::floor(xmin / spacing)),
      xmax(std::ceil(xmax / spacing)),
      ymin(std::floor(ymin / spacing)),
      ymax(std::ceil(ymax / spacing)),
      zmin(std::floor(zmin / spacing)),
      zmax(std::ceil(zmax / spacing)),
      spacing(spacing)
{
    initialize();
}

VisitationMap::~VisitationMap()
{
//    delete[] frequency_data;
}

void VisitationMap::initialize()
{
    //TODO: Look into fixing double to int conversion.
    width =  std::abs(xmin - xmax);
    height = std::abs(ymin - ymax);
    depth =  std::abs(zmin - zmax);

    GLint size;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &size);

    if(GetNumberOfBytes() > size)
    {
        throw std::runtime_error("Cannot construct visitation map as it does not fit into the available shader storage "
                                 "block size. Consider reducing NUMBER_OF_REPRESENTATIVE_FIBERS.");
    }

    //Visitation Map frequencies itself
    unsigned int numberOfCells = width * height * depth;
    cell_data = new Cell[numberOfCells];
    std::memset(cell_data, 0, sizeof(Cell) * numberOfCells);

    gl.glGenBuffers(1, &cells_ssbo);
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo);
//    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int) * width * height * depth, &frequency_data, GL_DYNAMIC_DRAW);
//    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo);
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind
}


unsigned int VisitationMap::getCellIndex(unsigned int x_index, unsigned int y_index, unsigned int z_index) const
{
    return x_index + width * (y_index + z_index * height);
}

void VisitationMap::getIndices(const glm::vec3& point, unsigned int& x_index, unsigned int& y_index, unsigned int& z_index) const
{
    //Casting to uint automatically floors the float
    x_index = uint((point.x - xmin * spacing) / spacing);
    y_index = uint((point.y - ymin * spacing) / spacing);
    z_index = uint((point.z - zmin * spacing) / spacing);
}

VisitationMap VisitationMap::CreateVisitationMapFromDWIDimensions(GL& gl,
                                                                  unsigned int nr_of_voxels_x,
                                                                  unsigned int nr_of_voxels_y,
                                                                  unsigned int nr_of_voxels_z,
                                                                  float dwi_spacing, float vm_spacing)
{
    auto x = (float)nr_of_voxels_x;
    auto y = (float)nr_of_voxels_y;
    auto z = (float)nr_of_voxels_z;

    float xmin = -(x * dwi_spacing) / 2.0f;
    float xmax =  (x * dwi_spacing) / 2.0f;
    float ymin = -(y * dwi_spacing) / 2.0f;
    float ymax =  (y * dwi_spacing) / 2.0f;
    float zmin = -(z * dwi_spacing) / 2.0f;
    float zmax =  (z * dwi_spacing) / 2.0f;

    return VisitationMap(gl, xmin, xmax, ymin, ymax, zmin, zmax, vm_spacing);
}
