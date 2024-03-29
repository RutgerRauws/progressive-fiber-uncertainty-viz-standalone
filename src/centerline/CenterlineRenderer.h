//
// Created by rutger on 8/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H


#include <functional>
#include <mutex>
#include "src/util/GL.h"
#include "../interaction/KeyPressObserver.h"
#include "../util/Fiber.h"
#include "../util/FiberObserver.h"
#include "../util/RenderElement.h"
#include "DistanceTablesUpdater.h"

class CenterlineRenderer : public FiberObserver, public RenderElement
{
    private:
        static constexpr auto VERTEX_SHADER_PATH   = "./shaders/centerline/vertex.glsl";
        static constexpr auto GEOMETRY_SHADER_PATH = "./shaders/centerline/geometry.glsl";
        static constexpr auto FRAGMENT_SHADER_PATH = "./shaders/centerline/fragment.glsl";

        GL& gl;

        unsigned int numberOfSeedPoints;

        GLint radiusLoc;
        GLint showCenterlineLoc;
        GLint colorLoc;
        GLint cameraPosLoc;

        std::vector<const Fiber*> centerFibers;

        const DistanceTableCollection& distanceTables;

        //Vertex storage
        unsigned int numberOfFibers;
        std::vector<float> verticesVector;
        std::vector<int> firstVertexOfEachFiber;
        std::vector<int> numberOfVerticesPerFiber;

        std::mutex mtx;

        void initialize() override;
        void updateData();
        void sendData();


public:
        CenterlineRenderer(GL& gl, const DistanceTableCollection& distanceTables, const Camera& camera);

        void NewFiber(Fiber* fiber) override;
        void Render() override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
