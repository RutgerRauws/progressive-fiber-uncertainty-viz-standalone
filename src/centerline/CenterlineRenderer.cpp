//
// Created by rutger on 8/1/20.
//

#include <Configuration.h>
#include "CenterlineRenderer.h"
#include "DistanceTablesUpdater.h"

CenterlineRenderer::CenterlineRenderer(GL& gl,
                                       const DistanceTableCollection& distanceTables,
                                       const Camera& camera)
    : RenderElement(VERTEX_SHADER_PATH, GEOMETRY_SHADER_PATH, FRAGMENT_SHADER_PATH, camera),
      gl(gl),
      distanceTables(distanceTables),
      numberOfSeedPoints(distanceTables.GetNumberOfSeedPoints()),
      showCenterlineLoc(-1),
      numberOfFibers(0)
{
    initialize();
}

void CenterlineRenderer::initialize()
{
    centerFibers.reserve(numberOfSeedPoints);

    for(unsigned int i = 0; i < numberOfSeedPoints; i++)
    {
        centerFibers.push_back(nullptr);
    }

    shaderProgram->bind();

    gl.glGenVertexArrays(1, &vao);
    gl.glGenBuffers(1, &vbo);

    //Get uniform locations
    modelMatLoc = gl.glGetUniformLocation(shaderProgram->programId(), "modelMat");
    viewMatLoc = gl.glGetUniformLocation(shaderProgram->programId(), "viewMat");
    projMatLoc = gl.glGetUniformLocation(shaderProgram->programId(), "projMat");

    cameraPosLoc = gl.glGetUniformLocation(shaderProgram->programId(), "cameraPosition");

    showCenterlineLoc = gl.glGetUniformLocation(shaderProgram->programId(), "showFibers");
}

void CenterlineRenderer::updateData()
{
    mtx.lock();

    verticesVector.clear();
    firstVertexOfEachFiber.clear();
    numberOfVerticesPerFiber.clear();

    for(const Fiber* fiber : centerFibers)
    {
        if(fiber == nullptr) { continue; }

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
    }

    vertices = verticesVector.data();
    numberOfFibers = firstVertexOfEachFiber.size();

    mtx.unlock();
}

void CenterlineRenderer::sendData()
{
    gl.glBindVertexArray(vao);

    gl.glBindBuffer(GL_ARRAY_BUFFER, vbo);
    gl.glBufferData(GL_ARRAY_BUFFER, GetNumberOfBytes(), GetVertexBufferData(), GL_DYNAMIC_DRAW); //TODO: there was a segfault here before, but not sure why
    gl.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    gl.glEnableVertexAttribArray(0);

    gl.glBindVertexArray(0);
}

void CenterlineRenderer::NewFiber(Fiber* fiber)
{
    const DistanceTable& distanceTable = distanceTables.GetDistanceTable(fiber->GetSeedPointId());
    const Fiber* currentCenterFiber = centerFibers.at(fiber->GetSeedPointId());

    if(currentCenterFiber == nullptr || distanceTable.GetCenterline().GetId() != currentCenterFiber->GetId())
    {
        centerFibers.at(fiber->GetSeedPointId()) = &distanceTable.GetCenterline();
        updateData();
    }
}

void CenterlineRenderer::Render()
{
    shaderProgram->bind();

    mtx.lock();
    sendData();

    gl.glBindVertexArray(vao);

    gl.glUniformMatrix4fv(modelMatLoc, 1, GL_FALSE, glm::value_ptr(camera.modelMatrix));
    gl.glUniformMatrix4fv(viewMatLoc, 1, GL_FALSE, glm::value_ptr(camera.viewMatrix));
    gl.glUniformMatrix4fv(projMatLoc, 1, GL_FALSE, glm::value_ptr(camera.projectionMatrix));

    gl.glUniform1i(showCenterlineLoc, Configuration::getInstance().SHOW_REPRESENTATIVE_FIBERS);

    gl.glUniform3f(cameraPosLoc, camera.cameraPos.x, camera.cameraPos.y, camera.cameraPos.z);

    gl.glMultiDrawArrays(GL_LINE_STRIP, &firstVertexOfEachFiber.front(), &numberOfVerticesPerFiber.front(), numberOfFibers);
    mtx.unlock();

    gl.glBindVertexArray(0);
}

unsigned int CenterlineRenderer::GetNumberOfVertices()
{
    return verticesVector.size() / 3;
}

unsigned int CenterlineRenderer::GetNumberOfBytes()
{
    return verticesVector.size() * sizeof(float);
}
