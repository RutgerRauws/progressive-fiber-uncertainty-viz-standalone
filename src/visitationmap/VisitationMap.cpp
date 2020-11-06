//
// Created by rutger on 10/15/20.
//

#include <algorithm>
#include <GL/glew.h>
#include <cmath>
#include <iostream>
#include <cstring>
#include "VisitationMap.h"
#include "../util/glm/glm.hpp"

VisitationMap::VisitationMap(GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax, GLfloat spacing)
    : xmin(std::floor(xmin / spacing)),
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

    glGenBuffers(1, &cells_ssbo);
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

VisitationMap VisitationMap::CreateTest()
{
    const float DTI_XMIN = -112;
    const float DTI_XMAX =  112;
    const float DTI_YMIN = -112;
    const float DTI_YMAX =  112;
    const float DTI_ZMIN = -70;
    const float DTI_ZMAX =  70;
//    const float DTI_SPACING = 0.25; //mm
//    const float DTI_SPACING = 0.75;
    const float DTI_SPACING = 1;

    const float MAX_FIBER_LENGTH = 190; //mm? ~186.861mm
    const glm::vec3 SEED_POINT(10.254, -6.92531, 0.630174); //estimate
//    DTI/DWI volume dimensions from example data set
//    const unsigned int width  = 112;
//    const unsigned int height = 112;
//    const unsigned int depth  = 70;
//    const float spacing = 2;
    VisitationMap visitationMap(DTI_XMIN, DTI_XMAX, DTI_YMIN, DTI_YMAX, DTI_ZMIN, DTI_ZMAX, DTI_SPACING);
//    visitationMap.makeSphere();
//    VisitationMap visitationMap = CreateVisitationMap(SEED_POINT, MAX_FIBER_LENGTH);

    return visitationMap;
}

VisitationMap VisitationMap::CreateVisitationMap(const glm::vec3& seedPoint, float cutoffLength)
{
    return VisitationMap(
            seedPoint.x - cutoffLength,
            seedPoint.x + cutoffLength,
            seedPoint.y - cutoffLength,
            seedPoint.y + cutoffLength,
            seedPoint.z - cutoffLength,
            seedPoint.z + cutoffLength,
            2
    );
}
