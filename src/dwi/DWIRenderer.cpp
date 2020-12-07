//
// Created by rutger on 10/8/20.
//
#include <algorithm>
#include <libs/glm/ext.hpp>
#include "DWIRenderer.h"

DWIRenderer::DWIRenderer(GL& gl, const Camera& camera, const DWISlice& slice, bool& showSlice)
    : RenderElement(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH, camera),
      gl(gl), slice(slice), showSlice(showSlice)
{
    initialize();
}

DWIRenderer::~DWIRenderer()
{
    delete[] vertices;
}

void DWIRenderer::initialize()
{
    vertices = slice.GetVertices();

    shaderProgram->bind();

    gl.glGenVertexArrays(1, &vao);
    gl.glGenBuffers(1, &vbo);

    gl.glBindVertexArray(vao);

    gl.glBindBuffer(GL_ARRAY_BUFFER, vbo);
    gl.glBufferData(GL_ARRAY_BUFFER, 8*4*sizeof(float), GetVertexBufferData(), GL_STATIC_DRAW);


    gl.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    gl.glEnableVertexAttribArray(0);

    gl.glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    gl.glEnableVertexAttribArray(1);

    GLint programId = shaderProgram->programId();

    modelMatLoc = gl.glGetUniformLocation(programId, "modelMat");
    viewMatLoc = gl.glGetUniformLocation(programId, "viewMat");
    projMatLoc = gl.glGetUniformLocation(programId, "projMat");

    gl.glGenTextures(1, &texture_loc);
    gl.glBindTexture(GL_TEXTURE_2D, texture_loc);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, slice.GetWidth(), slice.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, slice.GetBytes());
    gl.glGenerateMipmap(GL_TEXTURE_2D);

    opacity_loc = gl.glGetUniformLocation(programId, "showSlice");
    gl.glUniform1i(opacity_loc, showSlice);
}

void DWIRenderer::Render()
{
    shaderProgram->bind();

    gl.glBindTexture(GL_TEXTURE_2D, texture_loc);

    gl.glBindVertexArray(vao);

    gl.glUniformMatrix4fv(modelMatLoc, 1, GL_FALSE, glm::value_ptr(camera.modelMatrix));
    gl.glUniformMatrix4fv(viewMatLoc, 1, GL_FALSE, glm::value_ptr(camera.viewMatrix));
    gl.glUniformMatrix4fv(projMatLoc, 1, GL_FALSE, glm::value_ptr(camera.projectionMatrix));

    gl.glUniform1i(opacity_loc, showSlice);

    gl.glDrawArrays(GL_TRIANGLES, 0, 6);
}

unsigned int DWIRenderer::GetNumberOfVertices()
{
    return 36; //6 faces which each contain 6 vertices
}

unsigned int DWIRenderer::GetNumberOfBytes()
{
    return GetNumberOfVertices() * 5 * sizeof(float);
}