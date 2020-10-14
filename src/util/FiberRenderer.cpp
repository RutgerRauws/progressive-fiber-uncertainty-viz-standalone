//
// Created by rutger on 7/2/20.
//

#include <GL/glew.h>
#include "FiberRenderer.h"
#include "glm/ext.hpp"

FiberRenderer::FiberRenderer(const CameraState& cameraState)
    : RenderElement(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH, cameraState),
      numberOfFibers(0), fibersShown(true), pointsShown(false)
{
    initialize();
}

void FiberRenderer::initialize()
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    //todo: use shaderProgram->Use()?

    //Get uniform locations
    modelMatLoc = glGetUniformLocation(shaderProgram->GetId(), "modelMat");
    viewMatLoc = glGetUniformLocation(shaderProgram->GetId(), "viewMat");
    projMatLoc = glGetUniformLocation(shaderProgram->GetId(), "projMat");
}

void FiberRenderer::updateData()
{
    if(GetVertexBufferData() == nullptr)
    {
        return;
    }

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, GetNumberOfBytes(), GetVertexBufferData(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glEnableVertexAttribArray(0);
}

void FiberRenderer::NewFiber(Fiber* fiber)
{
    const std::vector<Point>& fiberPoints = fiber->GetPoints();

    unsigned int incomingNumberOfPoints = fiberPoints.size();
    unsigned int currentNumberOfPoints = GetNumberOfVertices();

    for(unsigned int i = 0; i < incomingNumberOfPoints; i++)
    {
        const Point& point = fiberPoints[i];

        verticesVector.push_back(point.X);
        verticesVector.push_back(point.Y);
        verticesVector.push_back(point.Z);
    }

    firstVertexOfEachFiber.push_back(currentNumberOfPoints);
    numberOfVerticesPerFiber.push_back(incomingNumberOfPoints);

    vertices = &verticesVector.front();
    numberOfFibers++;
    updateData();
}

void FiberRenderer::Render()
{
    shaderProgram->Use();

    updateData();
    glBindVertexArray(vao);

    glUniformMatrix4fv(modelMatLoc, 1, GL_FALSE, glm::value_ptr(cameraState.modelMatrix));
    glUniformMatrix4fv(viewMatLoc, 1, GL_FALSE, glm::value_ptr(cameraState.viewMatrix));
    glUniformMatrix4fv(projMatLoc, 1, GL_FALSE, glm::value_ptr(cameraState.projectionMatrix));

    glMultiDrawArrays(GL_LINE_STRIP, &firstVertexOfEachFiber.front(), &numberOfVerticesPerFiber.front(), numberOfFibers);
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
//    if(key == sf::Keyboard::P)
//    {
//        if(pointsShown)
//        {
//            renderer->RemoveActor(pointsActor);
//        }
//        else
//        {
//            renderer->AddActor(pointsActor);
//        }
//
//        pointsShown = !pointsShown;
//    }
//    else if(key == sf::Keyboard::F)
//    {
//        if(fibersShown)
//        {
//            renderer->RemoveActor(fiberActor);
//        }
//        else
//        {
//            renderer->AddActor(fiberActor);
//        }
//
//        fibersShown = !fibersShown;
//    }
}