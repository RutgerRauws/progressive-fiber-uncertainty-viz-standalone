//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H

#include <mutex>
#include "FiberObserver.h"
#include "../interaction/KeyPressObserver.h"
#include "RenderElement.h"

class FiberRenderer : public FiberObserver, public KeyPressObserver, RenderElement
{
    private:
        static constexpr auto VERTEX_SHADER_PATH   = "./shaders/fibers/vertex.glsl";
        static constexpr auto FRAGMENT_SHADER_PATH = "./shaders/fibers/fragment.glsl";

        //Fiber storage
        std::vector<float> verticesVector;
        std::vector<int> firstVertexOfEachFiber;
        unsigned int numberOfFibers;

        std::vector<int> numberOfVerticesPerFiber;

        std::mutex mtx; // mutex for critical section, NewFiber() is called from various threads

        GLint showFibersLoc;

        //Options
        bool showFibers, showPoints;

        void initialize() override;
        void updateData();

    public:
        explicit FiberRenderer(const Camera& camera);

        void NewFiber(Fiber* fiber) override;
        void Render() override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;

        void KeyPressed(const sf::Keyboard::Key& key) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H
