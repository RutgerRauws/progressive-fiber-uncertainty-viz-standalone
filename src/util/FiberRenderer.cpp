//
// Created by rutger on 7/2/20.
//

#include <GL/glew.h>
#include "FiberRenderer.h"

FiberRenderer::FiberRenderer()
    : fibersShown(true), pointsShown(false), numberOfFibers(0), numberOfVertices(0)
{
    FiberRenderer::initialize();
}

void FiberRenderer::initialize()
{
    vertices = nullptr;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

void FiberRenderer::NewFiber(Fiber* fiber)
{
    const std::vector<Point>& fiberPoints = fiber->GetPoints();

    float* oldPtr = GetVertexBufferData();
    unsigned int incomingNumberOfPoints = fiberPoints.size();
    unsigned int currentNumberOfPoints = GetNumberOfVertices();

    unsigned int newNumberOfPoints = currentNumberOfPoints + incomingNumberOfPoints;

    float* newVertices = new float[newNumberOfPoints * 3];

    if(currentNumberOfPoints > 0)
    {
        memcpy(newVertices, vertices, currentNumberOfPoints * 3);
    }

    for(unsigned int i = 0; i < fiberPoints.size(); i += 3)
    {
        const Point& point = fiberPoints[i];

        newVertices[currentNumberOfPoints * 3 + i + 0] = point.X;
        newVertices[currentNumberOfPoints * 3 + i + 1] = point.Y;
        newVertices[currentNumberOfPoints * 3 + i + 2] = point.Z;

//        std::cout << point.X << ", " << point.Y << ", " << point.Z << std::endl;
    }

    vertices = newVertices;
    numberOfVertices += incomingNumberOfPoints;
    numberOfFibers++;
    updateData();

    delete[] oldPtr;
//    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
//    polyLine->GetPointIds()->SetNumberOfIds(fiberPoints.size());
//
//    for(unsigned int i = 0; i < fiberPoints.size(); i++)
//    {
//        const Point& point = fiberPoints[i];
//
//        points->InsertPoint(currentId, point.X, point.Y, point.Z);
//        polyLine->GetPointIds()->SetId(i, currentId);
//        currentId++;
//    }
//
//    // Create a cell array to store the lines in and add the lines to it
//    polyLines->InsertNextCell(polyLine);
}

void FiberRenderer::Render()
{
    glBindVertexArray(vao);
    glDrawArrays(GL_LINES, 0, GetNumberOfVertices());
}

unsigned int FiberRenderer::GetNumberOfVertices()
{
    return numberOfVertices;
}

unsigned int FiberRenderer::GetNumberOfBytes()
{
    return GetNumberOfVertices() * 3 * sizeof(float);
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
