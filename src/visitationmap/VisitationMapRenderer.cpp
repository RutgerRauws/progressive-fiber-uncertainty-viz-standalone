//
// Created by rutger on 10/8/20.
//
#include <GL/glew.h>
#include <algorithm>
#include <Configuration.h>
#include "VisitationMapRenderer.h"

VisitationMapRenderer::VisitationMapRenderer(VisitationMap& visitationMap,
                                             RegionsOfInterest& regionsOfInterest,
                                             const DistanceTableCollection& distanceTables,
                                             const Camera& camera)
    : RenderElement(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH, camera),
      visitationMap(visitationMap),
      regionsOfInterest(regionsOfInterest),
      distanceTables(distanceTables),
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
    Configuration& config = Configuration::getInstance();

    shaderProgram->Use();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetSSBOId());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, visitationMap.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, regionsOfInterest.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, distanceTables.GetSSBOId());
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

    frequency_isovalue_loc = glGetUniformLocation(programId, "frequencyIsovalueThreshold");
    glProgramUniform1ui(programId, frequency_isovalue_loc, config.ISOVALUE_MIN_FREQUENCY_PERCENTAGE);

    distance_score_isovalue_loc = glGetUniformLocation(programId, "maxDistanceScoreIsovalueThreshold");
    glProgramUniform1d(programId, distance_score_isovalue_loc, config.ISOVALUE_MAX_DISTANCE_SCORE);

    use_frequency_isovalue_loc = glGetUniformLocation(programId, "useFrequencyIsovalue");
    glProgramUniform1i(programId, use_frequency_isovalue_loc, config.USE_FIBER_FREQUENCIES);

    use_interpolcation_loc = glGetUniformLocation(programId, "useInterpolation");
    glProgramUniform1i(programId, use_interpolcation_loc, config.USE_TRILINEAR_INTERPOLATION);
}

unsigned int VisitationMapRenderer::computeFrequencyIsovalue() const
{
    return std::ceil((float)numberOfFibers * Configuration::getInstance().ISOVALUE_MIN_FREQUENCY_PERCENTAGE);
}

void VisitationMapRenderer::Render()
{
    Configuration& config = Configuration::getInstance();

    shaderProgram->Use();
    glBindVertexArray(vao);

    glUniformMatrix4fv(modelMatLoc, 1, GL_FALSE, glm::value_ptr(camera.modelMatrix));
    glUniformMatrix4fv(viewMatLoc, 1, GL_FALSE, glm::value_ptr(camera.viewMatrix));
    glUniformMatrix4fv(projMatLoc, 1, GL_FALSE, glm::value_ptr(camera.projectionMatrix));

    glProgramUniform3f(shaderProgram->GetId(), cameraPos_loc, camera.cameraPos.x, camera.cameraPos.y, camera.cameraPos.z);

    glProgramUniform1i(shaderProgram->GetId(), use_frequency_isovalue_loc, config.USE_FIBER_FREQUENCIES);
    glProgramUniform1i(shaderProgram->GetId(), use_interpolcation_loc, config.USE_TRILINEAR_INTERPOLATION);
    glProgramUniform1ui(shaderProgram->GetId(), frequency_isovalue_loc, computeFrequencyIsovalue());
    glProgramUniform1d(shaderProgram->GetId(), distance_score_isovalue_loc, config.ISOVALUE_MAX_DISTANCE_SCORE);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());

    glDrawArrays(GL_TRIANGLES, 0, GetNumberOfVertices());
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