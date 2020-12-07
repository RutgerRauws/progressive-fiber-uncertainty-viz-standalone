//
// Created by rutger on 10/8/20.
//
#include <algorithm>
#include <Configuration.h>
#include "VisitationMapRenderer.h"

VisitationMapRenderer::VisitationMapRenderer(GL& gl,
                                             VisitationMap& visitationMap,
                                             RegionsOfInterest& regionsOfInterest,
                                             const DistanceTableCollection& distanceTables,
                                             const Camera& camera)
    : RenderElement(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH, camera),
      gl(gl),
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

    shaderProgram->bind();

    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetSSBOId());
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, visitationMap.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, regionsOfInterest.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, distanceTables.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    gl.glGenVertexArrays(1, &vao);
    gl.glGenBuffers(1, &vbo);

    gl.glBindVertexArray(vao);

    gl.glBindBuffer(GL_ARRAY_BUFFER, vbo);
    gl.glBufferData(GL_ARRAY_BUFFER, GetNumberOfBytes(), GetVertexBufferData(), GL_STATIC_DRAW);

    gl.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    gl.glEnableVertexAttribArray(0);

    GLint programId = shaderProgram->programId();

    //Get uniform locations
    modelMatLoc = gl.glGetUniformLocation(programId, "modelMat");
    viewMatLoc = gl.glGetUniformLocation(programId, "viewMat");
    projMatLoc = gl.glGetUniformLocation(programId, "projMat");

    GLint vmProp_loc;
    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.dataset_aabb.xmin");
    gl.glProgramUniform1i(programId, vmProp_loc, visitationMap.GetXmin());
    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.dataset_aabb.xmax");
    gl.glProgramUniform1i(programId, vmProp_loc, visitationMap.GetXmax());
    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.dataset_aabb.ymin");
    gl.glProgramUniform1i(programId, vmProp_loc, visitationMap.GetYmin());
    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.dataset_aabb.ymax");
    gl.glProgramUniform1i(programId, vmProp_loc, visitationMap.GetYmax());
    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.dataset_aabb.zmin");
    gl.glProgramUniform1i(programId, vmProp_loc, visitationMap.GetZmin());
    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.dataset_aabb.zmax");
    gl.glProgramUniform1i(programId, vmProp_loc, visitationMap.GetZmax());

    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.cellSize");
    gl.glProgramUniform1f(programId, vmProp_loc, visitationMap.GetSpacing());

    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.width");
    gl.glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetWidth());
    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.height");
    gl.glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetHeight());
    vmProp_loc = gl.glGetUniformLocation(programId, "vmp.depth");
    gl.glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetDepth());

    cameraPos_loc = gl.glGetUniformLocation(programId, "cameraPosition");

    frequency_isovalue_loc = gl.glGetUniformLocation(programId, "frequencyIsovalueThreshold");
    gl.glProgramUniform1f(programId, frequency_isovalue_loc, config.ISOVALUE_MIN_FREQUENCY_PERCENTAGE);

    distance_score_isovalue_loc = gl.glGetUniformLocation(programId, "maxDistanceScoreIsovalueThreshold");
    gl.glProgramUniform1d(programId, distance_score_isovalue_loc, config.ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE);

    use_frequency_isovalue_loc = gl.glGetUniformLocation(programId, "useFrequencyIsovalue");
    gl.glProgramUniform1i(programId, use_frequency_isovalue_loc, config.USE_FIBER_FREQUENCIES);

    use_interpolcation_loc = gl.glGetUniformLocation(programId, "useInterpolation");
    gl.glProgramUniform1i(programId, use_interpolcation_loc, config.USE_TRILINEAR_INTERPOLATION);

    opacity_loc = gl.glGetUniformLocation(programId, "opacity");
    gl.glProgramUniform1f(programId, opacity_loc, config.HULL_OPACITY);

    k_ambient_loc  = gl.glGetUniformLocation(programId, "k_ambient");
    k_diffuse_loc  = gl.glGetUniformLocation(programId, "k_diffuse");
    k_specular_loc = gl.glGetUniformLocation(programId, "k_specular");

    gl.glProgramUniform3f(programId, k_ambient_loc, config.HULL_COLOR_AMBIENT.red() / 255.0f, config.HULL_COLOR_AMBIENT.green() / 255.0f, config.HULL_COLOR_AMBIENT.blue() / 255.0f);
    gl.glProgramUniform3f(programId, k_diffuse_loc, config.HULL_COLOR_DIFFUSE.red() / 255.0f, config.HULL_COLOR_DIFFUSE.green() / 255.0f, config.HULL_COLOR_DIFFUSE.blue() / 255.0f);
    gl.glProgramUniform3f(programId, k_specular_loc,config.HULL_COLOR_SPECULAR.red() / 255.0f, config.HULL_COLOR_SPECULAR.green() / 255.0f, config.HULL_COLOR_SPECULAR.blue() / 255.0f);
}

float VisitationMapRenderer::computeFrequencyIsovalue() const
{
    return numberOfFibers * Configuration::getInstance().ISOVALUE_MIN_FREQUENCY_PERCENTAGE + 0.0001f;
}

double VisitationMapRenderer::computeDistanceScoreIsovalue() const
{
    return distanceTables.GetLargestDistanceScore() * Configuration::getInstance().ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE;
}

void VisitationMapRenderer::Render()
{
    Configuration& config = Configuration::getInstance();

    shaderProgram->bind();
    gl.glBindVertexArray(vao);

    gl.glUniformMatrix4fv(modelMatLoc, 1, GL_FALSE, glm::value_ptr(camera.modelMatrix));
    gl.glUniformMatrix4fv(viewMatLoc, 1, GL_FALSE, glm::value_ptr(camera.viewMatrix));
    gl.glUniformMatrix4fv(projMatLoc, 1, GL_FALSE, glm::value_ptr(camera.projectionMatrix));

    GLint programId = shaderProgram->programId();

    gl.glProgramUniform3f(programId, cameraPos_loc, camera.cameraPos.x, camera.cameraPos.y, camera.cameraPos.z);

    gl.glProgramUniform1i(programId, use_frequency_isovalue_loc, config.USE_FIBER_FREQUENCIES);
    gl.glProgramUniform1i(programId, use_interpolcation_loc, config.USE_TRILINEAR_INTERPOLATION);
    gl.glProgramUniform1f(programId, frequency_isovalue_loc, computeFrequencyIsovalue());
    gl.glProgramUniform1d(programId, distance_score_isovalue_loc, computeDistanceScoreIsovalue());

    gl.glProgramUniform1f(programId, opacity_loc, config.HULL_OPACITY);

    gl.glProgramUniform3f(programId, k_ambient_loc, config.HULL_COLOR_AMBIENT.red() / 255.0f, config.HULL_COLOR_AMBIENT.green() / 255.0f, config.HULL_COLOR_AMBIENT.blue() / 255.0f);
    gl.glProgramUniform3f(programId, k_diffuse_loc, config.HULL_COLOR_DIFFUSE.red() / 255.0f, config.HULL_COLOR_DIFFUSE.green() / 255.0f, config.HULL_COLOR_DIFFUSE.blue() / 255.0f);
    gl.glProgramUniform3f(programId, k_specular_loc,config.HULL_COLOR_SPECULAR.red() / 255.0f, config.HULL_COLOR_SPECULAR.green() / 255.0f, config.HULL_COLOR_SPECULAR.blue() / 255.0f);

    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());

    gl.glDrawArrays(GL_TRIANGLES, 0, GetNumberOfVertices());
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