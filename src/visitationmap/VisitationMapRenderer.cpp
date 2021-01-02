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

    use_frequency_isovalue_loc = gl.glGetUniformLocation(programId, "useFrequencyIsovalue");
    gl.glProgramUniform1i(programId, use_frequency_isovalue_loc, config.USE_FIBER_FREQUENCIES);

    use_interpolcation_loc = gl.glGetUniformLocation(programId, "useInterpolation");
    gl.glProgramUniform1i(programId, use_interpolcation_loc, config.USE_TRILINEAR_INTERPOLATION);

    //Hull related
    hull_isovalue_loc = gl.glGetUniformLocation(programId, "hullIsovalueThreshold");
    gl.glProgramUniform1f(programId, hull_isovalue_loc, config.HULL_ISOVALUE_MIN_FREQUENCY_PERCENTAGE);

    hull_opacity_loc = gl.glGetUniformLocation(programId, "hullOpacity");
    gl.glProgramUniform1f(programId, hull_opacity_loc, config.HULL_OPACITY);

    hull_k_ambient_loc  = gl.glGetUniformLocation(programId, "hullKAmbient");
    hull_k_diffuse_loc  = gl.glGetUniformLocation(programId, "hullKDiffuse");
    hull_k_specular_loc = gl.glGetUniformLocation(programId, "hullKSpecular");

    gl.glProgramUniform3f(programId, hull_k_ambient_loc, config.HULL_COLOR_AMBIENT.red() / 255.0f, config.HULL_COLOR_AMBIENT.green() / 255.0f, config.HULL_COLOR_AMBIENT.blue() / 255.0f);
    gl.glProgramUniform3f(programId, hull_k_diffuse_loc, config.HULL_COLOR_DIFFUSE.red() / 255.0f, config.HULL_COLOR_DIFFUSE.green() / 255.0f, config.HULL_COLOR_DIFFUSE.blue() / 255.0f);
    gl.glProgramUniform3f(programId, hull_k_specular_loc, config.HULL_COLOR_SPECULAR.red() / 255.0f, config.HULL_COLOR_SPECULAR.green() / 255.0f, config.HULL_COLOR_SPECULAR.blue() / 255.0f);

    //Silhouette related
    silhouette_isovalue_loc = gl.glGetUniformLocation(programId, "silhouetteIsovalueThreshold");
    gl.glProgramUniform1f(programId, silhouette_isovalue_loc, config.SILHOUETTE_ISOVALUE_MIN_FREQUENCY_PERCENTAGE);

    silhouette_opacity_loc = gl.glGetUniformLocation(programId, "silhouetteOpacity");
    gl.glProgramUniform1f(programId, silhouette_opacity_loc, config.SILHOUETTE_OPACITY);

    silhouette_color_loc = gl.glGetUniformLocation(programId, "silhouetteColor");
    gl.glProgramUniform3f(programId, silhouette_color_loc, config.SILHOUETTE_COLOR.red() / 255.0f, config.SILHOUETTE_COLOR.green() / 255.0f, config.SILHOUETTE_COLOR.blue() / 255.0f);
}

float VisitationMapRenderer::computeFrequencyIsovalue(bool isForHull) const
{
    float percentage;
    if(isForHull)
    {
        percentage = Configuration::getInstance().HULL_ISOVALUE_MIN_FREQUENCY_PERCENTAGE;
    }
    else
    {
        percentage = Configuration::getInstance().SILHOUETTE_ISOVALUE_MIN_FREQUENCY_PERCENTAGE;
    }

    float frequencyIsovalue;
    if(percentage == 0)
    {
        frequencyIsovalue = 0.01f; //we set the threshold to 0.01f in order for empty voxels not to light up.
    }
    else
    {
        frequencyIsovalue = (float)numberOfFibers * percentage;
    }

    return frequencyIsovalue;
}

float VisitationMapRenderer::computeDistanceScoreIsovalue(bool isForHull) const
{
    float percentage;

    if(isForHull)
    {
        percentage = Configuration::getInstance().HULL_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE;
    }
    else
    {
        percentage = Configuration::getInstance().SILHOUETTE_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE;
    }

    return distanceTables.GetDistanceScoreForPercentage(percentage);
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

    //Isovalue related
    if(config.USE_FIBER_FREQUENCIES)
    {
        gl.glProgramUniform1f(programId, hull_isovalue_loc, computeFrequencyIsovalue(true));
        gl.glProgramUniform1f(programId, silhouette_isovalue_loc, computeFrequencyIsovalue(false));
    }
    else
    {
        gl.glProgramUniform1f(programId, hull_isovalue_loc, computeDistanceScoreIsovalue(true));
        gl.glProgramUniform1f(programId, silhouette_isovalue_loc, computeDistanceScoreIsovalue(false));
    }

    //Hull related
    gl.glProgramUniform1f(programId, hull_opacity_loc, config.HULL_OPACITY);

    gl.glProgramUniform3f(programId, hull_k_ambient_loc, config.HULL_COLOR_AMBIENT.red() / 255.0f, config.HULL_COLOR_AMBIENT.green() / 255.0f, config.HULL_COLOR_AMBIENT.blue() / 255.0f);
    gl.glProgramUniform3f(programId, hull_k_diffuse_loc, config.HULL_COLOR_DIFFUSE.red() / 255.0f, config.HULL_COLOR_DIFFUSE.green() / 255.0f, config.HULL_COLOR_DIFFUSE.blue() / 255.0f);
    gl.glProgramUniform3f(programId, hull_k_specular_loc, config.HULL_COLOR_SPECULAR.red() / 255.0f, config.HULL_COLOR_SPECULAR.green() / 255.0f, config.HULL_COLOR_SPECULAR.blue() / 255.0f);

    //Silhouette related
    gl.glProgramUniform1f(programId, silhouette_opacity_loc, config.SILHOUETTE_OPACITY);
    gl.glProgramUniform3f(programId, silhouette_color_loc, config.SILHOUETTE_COLOR.red() / 255.0f, config.SILHOUETTE_COLOR.green() / 255.0f, config.SILHOUETTE_COLOR.blue() / 255.0f);

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