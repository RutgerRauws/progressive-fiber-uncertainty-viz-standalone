//
// Created by rutger on 10/8/20.
//

#include <GL/glew.h>
#include "Box.h"

Box::Box(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
    : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax)
{
    createVertices();
    Box::initialize();
}

Box::Box(float *bounds)
    : Box(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
{}

Box::~Box()
{
    delete[] vertices;
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
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, GetNumberOfBytes(), GetVertexBufferData(), GL_STATIC_DRAW);

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