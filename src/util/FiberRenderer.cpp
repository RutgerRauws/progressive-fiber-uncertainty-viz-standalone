//
// Created by rutger on 7/2/20.
//

#include "Configuration.h"
#include "FiberRenderer.h"
#include "glm/ext.hpp"

FiberRenderer::FiberRenderer(GL& gl, const Camera& camera)
    : RenderElement(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH, camera),
      gl(gl),
      numberOfFibers(0)
{
    initialize();
}

void FiberRenderer::initialize()
{
    shaderProgram->bind();

    gl.glGenVertexArrays(1, &vao);
    gl.glGenBuffers(1, &vbo);

    //Get uniform locations
    modelMatLoc = gl.glGetUniformLocation(shaderProgram->programId(), "modelMat");
    viewMatLoc  = gl.glGetUniformLocation(shaderProgram->programId(), "viewMat");
    projMatLoc  = gl.glGetUniformLocation(shaderProgram->programId(), "projMat");

    showFibersLoc = gl.glGetUniformLocation(shaderProgram->programId(), "showFibers");
}

void FiberRenderer::updateData()
{
    if(GetVertexBufferData() == nullptr)
    {
        return;
    }

    mtx.lock();
    gl.glBindVertexArray(vao);

    gl.glBindBuffer(GL_ARRAY_BUFFER, vbo);
    gl.glBufferData(GL_ARRAY_BUFFER, GetNumberOfBytes(), GetVertexBufferData(), GL_DYNAMIC_DRAW); //TODO: there was a segfault here before, but not sure why
    gl.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    gl.glEnableVertexAttribArray(0);

    gl.glBindVertexArray(0);

    mtx.unlock();
}

void FiberRenderer::NewFiber(Fiber* fiber)
{
    mtx.lock();

    const std::vector<glm::vec3>& fiberPoints = fiber->GetUniquePoints();

    unsigned int incomingNumberOfPoints = fiberPoints.size();
    unsigned int currentNumberOfPoints = GetNumberOfVertices();

    for(unsigned int i = 0; i < incomingNumberOfPoints; i++)
    {
        const glm::vec3& point = fiberPoints[i];

        verticesVector.push_back(point.x);
        verticesVector.push_back(point.y);
        verticesVector.push_back(point.z);
    }

    firstVertexOfEachFiber.push_back(currentNumberOfPoints);
    numberOfVerticesPerFiber.push_back(incomingNumberOfPoints);//TODO: there was a segfault here before, but not sure why perhaps add in mutex locks in the render method as last resort

    vertices = verticesVector.data();
    numberOfFibers++;

    mtx.unlock();
}

void FiberRenderer::Render()
{
    shaderProgram->bind();

    updateData();

    gl.glBindVertexArray(vao);

    gl.glUniformMatrix4fv(modelMatLoc, 1, GL_FALSE, glm::value_ptr(camera.modelMatrix));
    gl.glUniformMatrix4fv(viewMatLoc, 1, GL_FALSE, glm::value_ptr(camera.viewMatrix));
    gl.glUniformMatrix4fv(projMatLoc, 1, GL_FALSE, glm::value_ptr(camera.projectionMatrix));

    gl.glUniform1i(showFibersLoc, Configuration::getInstance().SHOW_FIBER_SAMPLES);

    gl.glMultiDrawArrays(GL_LINE_STRIP, firstVertexOfEachFiber.data(), numberOfVerticesPerFiber.data(), numberOfFibers);

    gl.glBindVertexArray(0);
}

unsigned int FiberRenderer::GetNumberOfVertices()
{
    return verticesVector.size() / 3;
}

unsigned int FiberRenderer::GetNumberOfBytes()
{
    return verticesVector.size() * sizeof(float);
}