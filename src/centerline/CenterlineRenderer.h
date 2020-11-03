//
// Created by rutger on 8/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H


#include <functional>
#include <mutex>
#include "DistanceTable.h"
#include "../interaction/KeyPressObserver.h"
#include "../util/Fiber.h"
#include "../util/FiberObserver.h"
#include "../util/RenderElement.h"

class CenterlineRenderer : public FiberObserver, public KeyPressObserver, RenderElement
{
    private:
        static constexpr auto VERTEX_SHADER_PATH   = "./shaders/centerline/vertex.glsl";
        static constexpr auto FRAGMENT_SHADER_PATH = "./shaders/centerline/fragment.glsl";

        unsigned int numberOfSeedPoints;

        GLint showCenterlineLoc;
        bool showCenterline;

        std::vector<DistanceTable> distanceTables;
        std::vector<const Fiber*> centerFibers;

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
        CenterlineRenderer(const CameraState& cameraState, unsigned int numberOfSeedPoints);

        void NewFiber(Fiber* fiber) override;
        void Render() override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;

        void KeyPressed(const sf::Keyboard::Key& key) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
