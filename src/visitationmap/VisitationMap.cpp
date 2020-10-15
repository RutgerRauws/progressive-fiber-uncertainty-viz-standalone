//
// Created by rutger on 10/15/20.
//

#include <algorithm>
#include <GL/glew.h>
#include <cmath>
#include <iostream>
#include "VisitationMap.h"
#include "../util/glm/vec3.hpp"
#include "../util/glm/glm.hpp"

VisitationMap::VisitationMap(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float spacing)
    : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax), spacing(spacing)
{
    initialize();
}


void VisitationMap::initialize()
{
    //TODO: Look into fixing double to int conversion.
    width =  std::ceil( std::abs(xmin - xmax) / spacing);
    height = std::ceil(std::abs(ymin - ymax) / spacing);
    depth =  std::ceil(std::abs(zmin - zmax) / spacing);

    //Visitation Map frequencies itself
    frequency_data = new unsigned int[width * height * depth];
    std::fill_n(frequency_data, width * height * depth, 0);

    glGenBuffers(1, &frequency_map_ssbo);
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
    x_index = uint((point.x - xmin) / spacing);
    y_index = uint((point.y - ymin) / spacing);
    z_index = uint((point.z - zmin) / spacing);
}

void VisitationMap::makeSphere()
{
    glm::vec3 centerPointWC(
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0
    );

    unsigned int indices[3];

    getIndices(centerPointWC, indices[0], indices[1], indices[2]);

//    unsigned int cellIndex = getCellIndex(indices[0], indices[1], indices[2]);

    //int sideSize = 30;
    float sideSize = std::min(width, std::min(height, depth)) / 2.0;

    for(float x = -sideSize; x < sideSize; x++)
    {
        for(float y = -sideSize; y < sideSize; y++)
        {
            for(float z = -sideSize; z < sideSize; z++)
            {
//                unsigned int cellIndex = getCellIndex(indices[0], indices[1], indices[2]);
//
//                unsigned int cellIndex =
//                    getCellIndex(indices[0] + x, indices[1] + y, indices[2] + z);

                glm::vec3 newPoint = centerPointWC + glm::vec3(x, y, z);
                if(glm::distance(centerPointWC, newPoint) > sideSize)
                {
                    continue;
                }

                unsigned int cellIndex =
                        getCellIndex(indices[0] + x, indices[1] + y, indices[2] + z);

                if(cellIndex > width * height * depth)
                {
                    std::cerr << "Splat out of bounds!" << std::endl;
                    continue;
                }

                frequency_data[cellIndex] = 9;
            }
        }
    }
}

VisitationMap VisitationMap::CreateTest()
{
    const float DTI_XMIN = -112;
    const float DTI_XMAX =  112;
    const float DTI_YMIN = -112;
    const float DTI_YMAX =  112;
    const float DTI_ZMIN = -70;
    const float DTI_ZMAX =  70;
    const float DTI_SPACING = 2; //mm

//    DTI/DWI volume dimensions from example data set
//    const unsigned int width  = 112;
//    const unsigned int height = 112;
//    const unsigned int depth  = 70;
//    const float spacing = 2;
    VisitationMap visitationMap(DTI_XMIN, DTI_XMAX, DTI_YMIN, DTI_YMAX, DTI_ZMIN, DTI_ZMAX, DTI_SPACING);
//    visitationMap.makeSphere();

    return visitationMap;
}
