//
// Created by rutger on 10/8/20.
//

#include <GL/glew.h>
#include <algorithm>
#include "VisitationMapRenderer.h"
#include "../util/glm/ext.hpp"

VisitationMapRenderer::VisitationMapRenderer(VisitationMap& visitationMap,
                                             RegionsOfInterest& regionsOfInterest,
                                             const CameraState& cameraState)
    : RenderElement(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH, cameraState),
      visitationMap(visitationMap),
      regionsOfInterest(regionsOfInterest),
      isovaluePercentage(0),
      numberOfFibers(0)
{
    createVertices();
    initialize();
}

VisitationMapRenderer::~VisitationMapRenderer()
{
    delete[] vertices;
}

void VisitationMapRenderer::createVertices() {
    float xmin = visitationMap.GetXmin() * visitationMap.GetSpacing();
    float ymin = visitationMap.GetYmin() * visitationMap.GetSpacing();
    float zmin = visitationMap.GetZmin() * visitationMap.GetSpacing();
    float xmax = visitationMap.GetXmax() * visitationMap.GetSpacing();
    float ymax = visitationMap.GetYmax() * visitationMap.GetSpacing();
    float zmax = visitationMap.GetZmax() * visitationMap.GetSpacing();

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

void VisitationMapRenderer::initialize()
{
    shaderProgram->Use();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetFrequencyMapSSBOId());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, visitationMap.GetFrequencyMapSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, regionsOfInterest.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, GetNumberOfBytes(), GetVertexBufferData(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    glEnableVertexAttribArray(0);

    //Get uniform locations
    modelMatLoc = glGetUniformLocation(shaderProgram->GetId(), "modelMat");
    viewMatLoc = glGetUniformLocation(shaderProgram->GetId(), "viewMat");
    projMatLoc = glGetUniformLocation(shaderProgram->GetId(), "projMat");

    //Visitation Map Properties
    GLint programId = shaderProgram->GetId();

    GLint vmProp_loc;
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.xmin");
    glProgramUniform1i(programId, vmProp_loc, visitationMap.GetXmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.xmax");
    glProgramUniform1i(programId, vmProp_loc, visitationMap.GetXmax());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.ymin");
    glProgramUniform1i(programId, vmProp_loc, visitationMap.GetYmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.ymax");
    glProgramUniform1i(programId, vmProp_loc, visitationMap.GetYmax());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.zmin");
    glProgramUniform1i(programId, vmProp_loc, visitationMap.GetZmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.zmax");
    glProgramUniform1i(programId, vmProp_loc, visitationMap.GetZmax());

    vmProp_loc = glGetUniformLocation(programId, "vmp.cellSize");
    glProgramUniform1f(programId, vmProp_loc, visitationMap.GetSpacing());

    vmProp_loc = glGetUniformLocation(programId, "vmp.width");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetWidth());
    vmProp_loc = glGetUniformLocation(programId, "vmp.height");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetHeight());
    vmProp_loc = glGetUniformLocation(programId, "vmp.depth");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetDepth());

    cameraPos_loc = glGetUniformLocation(programId, "cameraPosition");

    isovalue_loc = glGetUniformLocation(programId, "isovalueThreshold");
    glProgramUniform1f(programId, isovalue_loc, 0);
}

void VisitationMapRenderer::updateIsovaluePercentage(float delta)
{
    if(isovaluePercentage + delta >= 0.0f
    && isovaluePercentage + delta <= 1.0f)
    {
        isovaluePercentage += delta;
    }
}

float VisitationMapRenderer::computeIsovalue()
{
    std::cout << "Percentage at " << isovaluePercentage * 100 << "% and isovalue threshold at " << numberOfFibers * isovaluePercentage << std::endl;
    return numberOfFibers * isovaluePercentage;
}

void VisitationMapRenderer::Render()
{
    shaderProgram->Use();

    glBindVertexArray(vao);

    glUniformMatrix4fv(modelMatLoc, 1, GL_FALSE, glm::value_ptr(cameraState.modelMatrix));
    glUniformMatrix4fv(viewMatLoc, 1, GL_FALSE, glm::value_ptr(cameraState.viewMatrix));
    glUniformMatrix4fv(projMatLoc, 1, GL_FALSE, glm::value_ptr(cameraState.projectionMatrix));

    glProgramUniform3f(shaderProgram->GetId(), cameraPos_loc, cameraState.cameraPos.x, cameraState.cameraPos.y, cameraState.cameraPos.z);

    glProgramUniform1f(shaderProgram->GetId(), isovalue_loc, computeIsovalue());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetFrequencyMapSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());

    glDrawArrays(GL_TRIANGLES, 0, GetNumberOfVertices());
}

void VisitationMapRenderer::KeyPressed(const sf::Keyboard::Key &key)
{
    //Increase isovalue
    if(key == sf::Keyboard::U)
    {
        updateIsovaluePercentage(PERCENTAGE_DELTA);
    }

    //Decrease isovalue
    if(key == sf::Keyboard::J)
    {
        updateIsovaluePercentage(-PERCENTAGE_DELTA);
    }
}

void VisitationMapRenderer::NewFiber(Fiber *fiber)
{
    numberOfFibers++;
}


unsigned int VisitationMapRenderer::GetNumberOfVertices()
{
    return 36; //6 faces which each contain 6 vertices
}

unsigned int VisitationMapRenderer::GetNumberOfBytes()
{
    return GetNumberOfVertices() * 5 * sizeof(float);
}