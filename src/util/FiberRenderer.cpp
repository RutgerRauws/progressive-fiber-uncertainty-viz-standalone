//
// Created by rutger on 7/2/20.
//

#include <GL/glew.h>
#include "FiberRenderer.h"
#include "glm/ext.hpp"

FiberRenderer::FiberRenderer(const Camera& camera)
    : RenderElement(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH, camera),
      numberOfFibers(0), showFibers(false), showPoints(false)
{
    initialize();
}

void FiberRenderer::initialize()
{
    shaderProgram->Use();

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    //Get uniform locations
    modelMatLoc = glGetUniformLocation(shaderProgram->GetId(), "modelMat");
    viewMatLoc = glGetUniformLocation(shaderProgram->GetId(), "viewMat");
    projMatLoc = glGetUniformLocation(shaderProgram->GetId(), "projMat");

    showFibersLoc = glGetUniformLocation(shaderProgram->GetId(), "showFibers");
}

void FiberRenderer::updateData()
{
    if(GetVertexBufferData() == nullptr)
    {
        return;
    }

    mtx.lock();
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, GetNumberOfBytes(), GetVertexBufferData(), GL_DYNAMIC_DRAW); //TODO: there was a segfault here before, but not sure why
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

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
    shaderProgram->Use();

    updateData();

    glBindVertexArray(vao);

    glUniformMatrix4fv(modelMatLoc, 1, GL_FALSE, glm::value_ptr(camera.modelMatrix));
    glUniformMatrix4fv(viewMatLoc, 1, GL_FALSE, glm::value_ptr(camera.viewMatrix));
    glUniformMatrix4fv(projMatLoc, 1, GL_FALSE, glm::value_ptr(camera.projectionMatrix));

    glUniform1i(showFibersLoc, showFibers);

    glMultiDrawArrays(GL_LINE_STRIP, firstVertexOfEachFiber.data(), numberOfVerticesPerFiber.data(), numberOfFibers);

    glBindVertexArray(0);
}

unsigned int FiberRenderer::GetNumberOfVertices()
{
    return verticesVector.size() / 3;
}

unsigned int FiberRenderer::GetNumberOfBytes()
{
    return verticesVector.size() * sizeof(float);
}

void FiberRenderer::KeyPressed(const sf::Keyboard::Key& key)
{
    if(key == sf::Keyboard::F)
    {
        showFibers = !showFibers;
    }
}