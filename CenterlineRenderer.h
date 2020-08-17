//
// Created by rutger on 8/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H


#include <functional>
#include "KeyPressObserver.h"
#include "Fiber.h"
#include "DistanceTable.h"
#include "FiberObserver.h"

class CenterlineRenderer : public KeyPressObserver, public FiberObserver
{
    private:
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkActor> actor;

        bool centerlineShown;

        DistanceTable distanceTable;
        const Fiber* centerfiber_ptr;

        void render();

    public:
        explicit CenterlineRenderer(vtkSmartPointer<vtkRenderer> renderer);

        void NewFiber(Fiber* fiber) override;

        void KeyPressed(const std::basic_string<char>& value) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
