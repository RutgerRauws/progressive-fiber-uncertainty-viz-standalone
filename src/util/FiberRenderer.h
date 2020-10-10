//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H

#include "FiberObserver.h"
#include "../interaction/KeyPressObserver.h"
#include "RenderElement.h"
#include "glm/vec3.hpp"

class FiberRenderer : public FiberObserver, public KeyPressObserver, RenderElement
{
    private:
        std::vector<float> verticesVector;
        std::vector<int> firstVertexOfEachFiber;
        unsigned int numberOfFibers;

        std::vector<int> numberOfVerticesPerFiber;

        bool fibersShown, pointsShown;

        void initialize() override;
        void updateData();

    public:
        explicit FiberRenderer();
        void NewFiber(Fiber* fiber) override;

        void Render() override;

        unsigned int GetNumberOfVertices() override;
        unsigned int GetNumberOfBytes() override;

        void KeyPressed(const sf::Keyboard::Key& key) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_RENDERER_H
