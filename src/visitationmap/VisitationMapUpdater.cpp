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

    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.xmin");
    glProgramUniform1f(programId, vmProp_loc, visitationMap.GetXmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.xmax");
    glProgramUniform1f(programId, vmProp_loc, visitationMap.GetXmax());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.ymin");
    glProgramUniform1f(programId, vmProp_loc, visitationMap.GetYmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.ymax");
    glProgramUniform1f(programId, vmProp_loc, visitationMap.GetYmax());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.zmin");
    glProgramUniform1f(programId, vmProp_loc, visitationMap.GetZmin());
    vmProp_loc = glGetUniformLocation(programId, "vmp.dataset_aabb.zmax");
    glProgramUniform1f(programId, vmProp_loc, visitationMap.GetZmax());

    vmProp_loc = glGetUniformLocation(programId, "vmp.cellSize");
    glProgramUniform1f(programId, vmProp_loc, visitationMap.GetSpacing());

    vmProp_loc = glGetUniformLocation(programId, "vmp.width");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetWidth());
    vmProp_loc = glGetUniformLocation(programId, "vmp.height");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetHeight());
    vmProp_loc = glGetUniformLocation(programId, "vmp.depth");
    glProgramUniform1ui(programId, vmProp_loc, visitationMap.GetDepth());

    //Visitation Map frequencies itself
    GLuint frequency_map_ssbo_id = visitationMap.GetSSBOId();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(visitationMap.GetAABB()) + visitationMap.GetNumberOfBytes(), 0, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(visitationMap.GetAABB()), (float*)&visitationMap.GetAABB());
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(visitationMap.GetAABB()), visitationMap.GetNumberOfBytes(), visitationMap.GetData());
//    glBufferData(GL_SHADER_STORAGE_BUFFER, visitationMap.GetNumberOfBytes(), visitationMap.GetData(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glGenBuffers(1, &fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fiber_segments_ssbo_id);
//    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, 0, )
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

//    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo_id);
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo_id);


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

    std::vector<glm::vec4> segmentsVertices;
    fiberQueueToSegmentVertices(segmentsVertices);

    shaderProgram->Use();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fiber_segments_ssbo_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * segmentsVertices.size(), segmentsVertices.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetSSBOId());

    int numberOfPoints = 20;
    int numberOfEdges = numberOfPoints - 1;

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void VisitationMapUpdater::fiberQueueToSegmentVertices(std::vector<glm::vec4>& outVertices)
{
    std::vector<Fiber*> fibersCopy(fiberQueue);
    fiberQueue.clear();

    for(Fiber* fiber : fibersCopy)
    {
        const std::vector<glm::vec4>& fiberPoints = fiber->GetPoints();
        outVertices.insert(outVertices.end(), fiberPoints.begin(), fiberPoints.end());
    }
}
