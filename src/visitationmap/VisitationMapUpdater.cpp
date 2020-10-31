//
// Created by rutger on 10/1/20.
//

#include "VisitationMapUpdater.h"
#include <iostream>

VisitationMapUpdater::VisitationMapUpdater(VisitationMap& visitationMap, RegionsOfInterest& regionsOfInterest)
   : visitationMap(visitationMap), regionsOfInterest(regionsOfInterest)
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

    //Visitation Map frequencies itself
    GLuint frequency_map_ssbo_id = visitationMap.GetFrequencyMapSSBOId();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo_id);
//    glBufferData(GL_SHADER_STORAGE_BUFFER, visitationMap.GetNumberOfBytes(), 0, GL_DYNAMIC_DRAW);
//    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(visitationMap.GetAABB()), (GLint*)&visitationMap.GetAABB());
//    glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(visitationMap.GetAABB()), visitationMap.GetNumberOfBytes(), visitationMap.GetData());
    glBufferData(GL_SHADER_STORAGE_BUFFER, visitationMap.GetNumberOfBytes(), visitationMap.GetData(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    regionsOfInterest.Initialize();

    glGenBuffers(1, &fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fiber_segments_ssbo_id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    visitationMap.GetCellFiberMultiMap().Initialize();

    //Get the limitations on the number of work groups the GPU supports
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxNrOfWorkGroups);
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

    std::vector<Fiber::LineSegment> segments;
    fiberQueueToSegmentVertices(segments);

    shaderProgram->Use();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fiber_segments_ssbo_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Fiber::LineSegment) * segments.size(), segments.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetFrequencyMapSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetCellFiberMultiMap().GetSSBOId());

    int numberOfLineSegments = segments.size();
    int numberOfWorkGroups = numberOfLineSegments;
    //int numberOfWorkGroups = std::min(numberOfEdges, maxNrOfWorkGroups); //TODO: we do not want to dispatch more workgroups than the GPU supports
    //minimum supported is 65535

    glDispatchCompute(numberOfWorkGroups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void VisitationMapUpdater::fiberQueueToSegmentVertices(std::vector<Fiber::LineSegment>& outSegments)
{
    std::vector<Fiber*> fibersCopy(fiberQueue);
    fiberQueue.clear();

    for(Fiber* fiber : fibersCopy)
    {
        const std::vector<Fiber::LineSegment>& lineSegments = fiber->GetLineSegments();
        outSegments.insert(outSegments.end(), lineSegments.begin(), lineSegments.end());
    }
}
