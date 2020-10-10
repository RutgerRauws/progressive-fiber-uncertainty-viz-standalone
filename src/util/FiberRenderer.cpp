//
// Created by rutger on 7/2/20.
//

#include <GL/glew.h>
#include <cstring>
#include "FiberRenderer.h"

FiberRenderer::FiberRenderer()
    : fibersShown(true), pointsShown(false), numberOfFibers(0)
{
    FiberRenderer::initialize();
}

void FiberRenderer::initialize()
{
    vertices = nullptr;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);


//    const float DTI_XMAX =  112;
//    const float DTI_YMAX =  112;
//    const float DTI_ZMAX =  70;
//
//
//    Fiber fiber(0);
//    fiber.AddPoint(0, 0, 0);
//    fiber.AddPoint(DTI_XMAX, DTI_YMAX, DTI_ZMAX);
//    fiber.AddPoint(DTI_XMAX, 2*DTI_YMAX, DTI_ZMAX);
//    fiber.AddPoint(2*DTI_XMAX, 2*DTI_YMAX, DTI_ZMAX);
//
//    const std::vector<Point>& fiberPoints = fiber.GetPoints();
//
//    unsigned int incomingNumberOfPoints = fiberPoints.size();
//    unsigned int currentNumberOfPoints = GetNumberOfVertices();
//
//    unsigned int newNumberOfPoints = currentNumberOfPoints + incomingNumberOfPoints;
//
//    float* newVertices = new float[newNumberOfPoints * 3];
//
//    if(currentNumberOfPoints > 0)
//    {
//        memcpy(newVertices, vertices, currentNumberOfPoints * 3);
//    }
//
//    for(unsigned int i = 0; i < fiberPoints.size(); i++)
//    {
//        const Point& point = fiberPoints[i];
//
//        newVertices[(currentNumberOfPoints + i) * 3 + 0] = point.X;
//        newVertices[(currentNumberOfPoints + i) * 3 + 1] = point.Y;
//        newVertices[(currentNumberOfPoints + i) * 3 + 2] = point.Z;
//
////        std::cout << point.X << ", " << point.Y << ", " << point.Z << std::endl;
//    }
//
//    vertices = newVertices;
//    numberOfVertices += incomingNumberOfPoints;
//    numberOfFibers++;
//    updateData();
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
    updateData();
    glBindVertexArray(vao);
//    glDrawArrays(GL_LINES, 0, GetNumberOfVertices());
//    glDrawArrays(GL_LINE_STRIP, 0, GetNumberOfVertices());
//    glDrawArrays()
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
