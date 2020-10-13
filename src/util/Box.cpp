//
// Created by rutger on 10/8/20.
//

#include <GL/glew.h>
#include <cmath>
#include <algorithm>
#include "Box.h"

Box::Box(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
    : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax)
{
    width =  std::ceil( std::abs(xmin - xmax) / spacing);
    height = std::ceil(std::abs(ymin - ymax) / spacing);
    depth =  std::ceil(std::abs(zmin - zmax) / spacing);

    createVertices();
    Box::initialize();
}

Box::Box(float *bounds)
    : Box(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
{}

Box::~Box()
{
    delete[] vertices;
    delete[] frequency_data;
}

void Box::createVertices() {
    vertices = new float[36 * 5] {
        xmin, ymin, zmin,  0.0f, 0.0f,
        xmax, ymin, zmin,  1.0f, 0.0f,
        xmax, ymax, zmin,  1.0f, 1.0f,
        xmax, ymax, zmin,  1.0f, 1.0f,
        xmin, ymax, zmin,  0.0f, 1.0f,
        xmin, ymin, zmin,  0.0f, 0.0f,

        xmin, ymin, zmax,  0.0f, 0.0f,
        xmax, ymin, zmax,  1.0f, 0.0f,
        xmax, ymax, zmax,  1.0f, 1.0f,
        xmax, ymax, zmax,  1.0f, 1.0f,
        xmin, ymax, zmax,  0.0f, 1.0f,
        xmin, ymin, zmax,  0.0f, 0.0f,

        xmin, ymax, zmax,  1.0f, 0.0f,
        xmin, ymax, zmin,  1.0f, 1.0f,
        xmin, ymin, zmin,  0.0f, 1.0f,
        xmin, ymin, zmin,  0.0f, 1.0f,
        xmin, ymin, zmax,  0.0f, 0.0f,
        xmin, ymax, zmax,  1.0f, 0.0f,

        xmax, ymax, zmax,  1.0f, 0.0f,
        xmax, ymax, zmin,  1.0f, 1.0f,
        xmax, ymin, zmin,  0.0f, 1.0f,
        xmax, ymin, zmin,  0.0f, 1.0f,
        xmax, ymin, zmax,  0.0f, 0.0f,
        xmax, ymax, zmax,  1.0f, 0.0f,

        xmin, ymin, zmin,  0.0f, 1.0f,
        xmax, ymin, zmin,  1.0f, 1.0f,
        xmax, ymin, zmax,  1.0f, 0.0f,
        xmax, ymin, zmax,  1.0f, 0.0f,
        xmin, ymin, zmax,  0.0f, 0.0f,
        xmin, ymin, zmin,  0.0f, 1.0f,

        xmin, ymax, zmin,  0.0f, 1.0f,
        xmax, ymax, zmin,  1.0f, 1.0f,
        xmax, ymax, zmax,  1.0f, 0.0f,
        xmax, ymax, zmax,  1.0f, 0.0f,
        xmin, ymax, zmax,  0.0f, 0.0f,
        xmin, ymax, zmin,  0.0f, 1.0f
    };
}

void Box::initialize()
{

    //Visitation Map frequencies itself
    frequency_data = new unsigned int[width * height * depth];
    std::fill_n(frequency_data, width * height * depth, 5);

    GLuint frequency_map_ssbo;
    glGenBuffers(1, &frequency_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int) * width * height * depth, frequency_data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind










    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, GetNumberOfBytes(), GetVertexBufferData(), GL_STATIC_DRAW);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    glEnableVertexAttribArray(0);
}

void Box::Render()
{
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, GetNumberOfVertices());
}

unsigned int Box::GetNumberOfVertices()
{
    return 36; //6 faces which each contain 6 vertices
}

unsigned int Box::GetNumberOfBytes()
{
    return GetNumberOfVertices() * 5 * sizeof(float);
}

void Box::SetUpUniforms(GLuint program)
{
    //Visitation Map Properties
    GLint vmProp_loc;
    vmProp_loc = glGetUniformLocation(program, "vmp.xmin");
    glProgramUniform1d(program, vmProp_loc, xmin);
    vmProp_loc = glGetUniformLocation(program, "vmp.xmax");
    glProgramUniform1d(program, vmProp_loc, xmax);
    vmProp_loc = glGetUniformLocation(program, "vmp.ymin");
    glProgramUniform1d(program, vmProp_loc, ymin);
    vmProp_loc = glGetUniformLocation(program, "vmp.ymax");
    glProgramUniform1d(program, vmProp_loc, ymax);
    vmProp_loc = glGetUniformLocation(program, "vmp.zmin");
    glProgramUniform1d(program, vmProp_loc, zmin);
    vmProp_loc = glGetUniformLocation(program, "vmp.zmax");
    glProgramUniform1d(program, vmProp_loc, zmax);

    vmProp_loc = glGetUniformLocation(program, "vmp.cellSize");
    glProgramUniform1d(program, vmProp_loc, spacing);

    vmProp_loc = glGetUniformLocation(program, "vmp.width");
    glProgramUniform1ui(program, vmProp_loc, width);
    vmProp_loc = glGetUniformLocation(program, "vmp.height");
    glProgramUniform1ui(program, vmProp_loc, height);
    vmProp_loc = glGetUniformLocation(program, "vmp.depth");
    glProgramUniform1ui(program, vmProp_loc, depth);
}