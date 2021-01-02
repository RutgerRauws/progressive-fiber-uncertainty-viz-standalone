//
// Created by rutger on 10/1/20.
//

#include "VisitationMapUpdater.h"
#include <iostream>
#include <QtGui/QOpenGLShader>

VisitationMapUpdater::VisitationMapUpdater(GL& gl,
                                           VisitationMap& visitationMap,
                                           RegionsOfInterest& regionsOfInterest,
                                           const DistanceTableCollection& distanceTables)
   : gl(gl),
     visitationMap(visitationMap),
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

    //Visitation Map frequencies itself
    GLuint visitation_map_ssbo_id = visitationMap.GetSSBOId();

    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitation_map_ssbo_id);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER, visitationMap.GetNumberOfBytes(), visitationMap.GetData(), GL_DYNAMIC_DRAW);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, visitation_map_ssbo_id);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    delete[] visitationMap.GetData(); //reduce internal memory load

    regionsOfInterest.Initialize();

    gl.glGenBuffers(1, &fiber_segments_ssbo_id);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, fiber_segments_ssbo_id);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fiber_segments_ssbo_id);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    std::vector<double> distanceScores = distanceTables.GetDistanceScoreCopy();
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER, distanceScores.size() * sizeof(double), distanceScores.data(), GL_DYNAMIC_DRAW);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, distanceTables.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    //Get the limitations on the number of work groups the GPU supports
    gl.glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxNrOfWorkGroups);
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

    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, fiber_segments_ssbo_id);
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Fiber::LineSegment) * segments.size(), segments.data(), GL_DYNAMIC_DRAW);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fiber_segments_ssbo_id);
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    std::vector<double> distanceScores = distanceTables.GetDistanceScoreCopy();
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());
    gl.glBufferData(GL_SHADER_STORAGE_BUFFER, distanceScores.size() * sizeof(double), distanceScores.data(), GL_DYNAMIC_DRAW);
    gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, distanceTables.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, visitationMap.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, regionsOfInterest.GetSSBOId());
    gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, distanceTables.GetSSBOId());

    int numberOfLineSegments = segments.size();
    int numberOfWorkGroups = numberOfLineSegments;
    //int numberOfWorkGroups = std::min(numberOfEdges, maxNrOfWorkGroups); //TODO: we do not want to dispatch more workgroups than the GPU supports
    //minimum supported is 65535

    gl.glDispatchCompute(
        std::ceil(visitationMap.GetWidth()  / 8.0f),
        std::ceil(visitationMap.GetHeight() / 8.0f),
        std::ceil(visitationMap.GetDepth()  / 8.0f)
    );

    gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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
