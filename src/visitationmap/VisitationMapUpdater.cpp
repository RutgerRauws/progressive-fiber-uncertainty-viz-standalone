//
// Created by rutger on 10/1/20.
//

#include "VisitationMapUpdater.h"

#include <sstream>
#include <iostream>
#include <cstring>
#include <fstream>

VisitationMapUpdater::VisitationMapUpdater(VisitationMap& visitationMap)
   : visitationMap(visitationMap)
{
    initialize();
}

void VisitationMapUpdater::initialize()
{
    try
    {
        computeShader = Shader::LoadFromFile(COMPUTE_SHADER_PATH, GL_COMPUTE_SHADER);
        computeShader->Compile();
    }
    catch(const ShaderError& e)
    {
        std::cerr << "Could not compile compute shader: " << e.what() << std::endl;
        throw e;
    }

    Shader* shaders[1] = {computeShader};
    shaderProgram = new ShaderProgram(shaders, 1);

    shaderProgram->Use();

    /***
     *
     * Preparing inputs
     *
     */
    //Visitation Map Properties
    GLint vmProp_loc;
    GLuint programId = shaderProgram->GetId();

    vmProp_loc = glGetUniformLocation(programId, "vmp.xmin");
    glProgramUniform1d(programId, vmProp_loc, visitationMap.GetXmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.xmax");
    glProgramUniform1d(programId, vmProp_loc, visitationMap.GetXmax());
    vmProp_loc = glGetUniformLocation(programId, "vmp.ymin");
    glProgramUniform1d(programId, vmProp_loc, visitationMap.GetYmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.ymax");
    glProgramUniform1d(programId, vmProp_loc, visitationMap.GetYmax());
    vmProp_loc = glGetUniformLocation(programId, "vmp.zmin");
    glProgramUniform1d(programId, vmProp_loc, visitationMap.GetZmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.zmax");
    glProgramUniform1d(programId, vmProp_loc, visitationMap.GetZmax());

    vmProp_loc = glGetUniformLocation(programId, "vmp.cellSize");
    glProgramUniform1d(programId, vmProp_loc, visitationMap.GetSpacing());

    vmProp_loc = glGetUniformLocation(programId, "vmp.width");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetWidth());
    vmProp_loc = glGetUniformLocation(programId, "vmp.height");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetHeight());
    vmProp_loc = glGetUniformLocation(programId, "vmp.depth");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetDepth());

    //Visitation Map frequencies itself
    GLuint frequency_map_ssbo = visitationMap.GetSSBOId();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int) * visitationMap.GetWidth() * visitationMap.GetHeight() * visitationMap.GetDepth(), visitationMap.GetData(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

//    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo);
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo);


//    glDispatchCompute(1, 1, 1); //(nr-of-segments / local-group-size, 1, 1)

    //Sync here to make writes visible
    //glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    //glMemoryBarrier(GL_COMPUTE_SHADER_BIT);
//    NewFiber(nullptr);
}

void VisitationMapUpdater::NewFiber(Fiber* fiber)
{
    fiberQueue.push_back(fiber);
}

void VisitationMapUpdater::Update()
{
    if(fiberQueue.empty())
    {
        return;
    }

    shaderProgram->Use();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetSSBOId());

    int numberOfPoints = 20;
    int numberOfEdges = numberOfPoints - 1;

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    fiberQueue.clear();
}