//
// Created by rutger on 8/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H


#include <functional>
#include <vtkRenderer.h>
#include <vtkActor.h>
#include "../interaction/KeyPressObserver.h"
#include "../util/Fiber.h"
#include "DistanceTable.h"
#include "../util/FiberObserver.h"

class CenterlineRenderer : public KeyPressObserver, public FiberObserver
{
    private:
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkActor> actor;

        bool centerlineShown;

        DistanceTable distanceTable;
        unsigned int centerFiberId;

        void render();

    public:
        explicit CenterlineRenderer(vtkSmartPointer<vtkRenderer> renderer);

        void NewFiber(Fiber* fiber) override;

        void KeyPressed(const sf::Keyboard::Key& key) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
