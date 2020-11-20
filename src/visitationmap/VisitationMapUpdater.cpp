//
// Created by rutger on 10/1/20.
//

#include "VisitationMapUpdater.h"
#include <iostream>
#include <QtGui/QOpenGLShader>

VisitationMapUpdater::VisitationMapUpdater(VisitationMap& visitationMap,
                                           RegionsOfInterest& regionsOfInterest,
                                           const DistanceTableCollection& distanceTables)
   : visitationMap(visitationMap),
     regionsOfInterest(regionsOfInterest),
     distanceTables(distanceTables)
{
    initialize();
}

VisitationMapUpdater::~VisitationMapUpdater()
{
    delete shaderProgram;
}

void VisitationMapUpdater::initialize()
{
    QOpenGLShader* computeShader = nullptr;

    computeShader = new QOpenGLShader(QOpenGLShader::ShaderTypeBit::Compute);
    computeShader->compileSourceFile(QString(COMPUTE_SHADER_PATH.data()));

    shaderProgram = new QOpenGLShaderProgram();
    shaderProgram->addShader(computeShader);

    shaderProgram->link();

    delete computeShader; //it's not used anymore after compiling

    shaderProgram->bind();

    /***
     *
     * Preparing inputs
     *
     */
    //Visitation Map Properties
    GLint vmProp_loc;
    GLuint programId = shaderProgram->programId();

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
    GLuint visitation_map_ssbo_id = visitationMap.GetSSBOId();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitation_map_ssbo_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, visitationMap.GetNumberOfBytes(), visitationMap.GetData(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, visitation_map_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    delete[] visitationMap.GetData(); //reduce internal memory load

    regionsOfInterest.Initialize();

    glGenBuffers(1, &fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fiber_segments_ssbo_id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    std::vector<double> distanceScores = distanceTables.GetDistanceScoreCopy();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());
    glBufferData(GL_SHADER_STORAGE_BUFFER, distanceScores.size() * sizeof(double), distanceScores.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, distanceTables.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    //Get the limitations on the number of work groups the GPU supports
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxNrOfWorkGroups);
}

void VisitationMapUpdater::NewFiber(Fiber* fiber)
{
    queueLock.lock();
    fiberQueue.push_back(fiber);
    queueLock.unlock();
}

static int nextPowerOfTwo(int x) {
    x--;
    x |= x >> 1; // handle 2 bit numbers
    x |= x >> 2; // handle 4 bit numbers
    x |= x >> 4; // handle 8 bit numbers
    x |= x >> 8; // handle 16 bit numbers
    x |= x >> 16; // handle 32 bit numbers
    x++;
    return x;
}

void VisitationMapUpdater::Update()
{
    if(fiberQueue.empty())
    {
        return;
    }

    std::vector<Fiber::LineSegment> segments;
    fiberQueueToSegmentVertices(segments);

    shaderProgram->bind();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fiber_segments_ssbo_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Fiber::LineSegment) * segments.size(), segments.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fiber_segments_ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    std::vector<double> distanceScores = distanceTables.GetDistanceScoreCopy();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());
    glBufferData(GL_SHADER_STORAGE_BUFFER, distanceScores.size() * sizeof(double), distanceScores.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, distanceTables.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());

    int numberOfLineSegments = segments.size();
    int numberOfWorkGroups = numberOfLineSegments;
    //int numberOfWorkGroups = std::min(numberOfEdges, maxNrOfWorkGroups); //TODO: we do not want to dispatch more workgroups than the GPU supports
    //minimum supported is 65535

    glDispatchCompute(
            nextPowerOfTwo(visitationMap.GetWidth() / 8),
            nextPowerOfTwo(visitationMap.GetHeight() / 8),
            nextPowerOfTwo(visitationMap.GetDepth() / 8)
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void VisitationMapUpdater::fiberQueueToSegmentVertices(std::vector<Fiber::LineSegment>& outSegments)
{
    queueLock.lock();
    std::vector<Fiber*> fibersCopy(fiberQueue);
    fiberQueue.clear();
    queueLock.unlock();

    for(Fiber* fiber : fibersCopy)
    {
        const std::vector<Fiber::LineSegment>& lineSegments = fiber->GetLineSegments();
        outSegments.insert(outSegments.end(), lineSegments.begin(), lineSegments.end());
    }
}
