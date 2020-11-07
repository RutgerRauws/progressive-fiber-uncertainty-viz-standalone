//
// Created by rutger on 10/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H

#include <string>
#include <GL/glew.h>
#include <mutex>
#include "VisitationMap.h"
#include "../util/Shader.h"
#include "../util/ShaderProgram.h"
#include "../util/FiberObserver.h"
#include "../centerline/DistanceTableCollection.h"

class VisitationMapUpdater : public FiberObserver
{
    private:
        const std::string COMPUTE_SHADER_PATH = "./shaders/visitationmap/compute.glsl";

        ShaderProgram* shaderProgram = nullptr;

        VisitationMap& visitationMap;
        RegionsOfInterest& regionsOfInterest;
        const DistanceTableCollection& distanceTables;

        std::mutex queueLock;
        std::vector<Fiber*> fiberQueue;

        GLuint fiber_segments_ssbo_id;

        GLint maxNrOfWorkGroups;


        void initialize();
        void fiberQueueToSegmentVertices(std::vector<Fiber::LineSegment>& outSegments);

    public:
        VisitationMapUpdater(VisitationMap& visitationMap,
                             RegionsOfInterest& regionsOfInterest,
                             const DistanceTableCollection& distanceTables);
        ~VisitationMapUpdater();

        void NewFiber(Fiber* fiber) override;
        void Update();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_VISITATION_MAP_UPDATER_H
