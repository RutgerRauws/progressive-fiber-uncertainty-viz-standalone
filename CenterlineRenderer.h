//
// Created by rutger on 8/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H


#include <functional>
#include "KeyPressObserver.h"
#include "Fiber.h"

class CenterlineRenderer : public KeyPressObserver
{
    private:
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkActor> actor;

        bool centerlineShown;

    public:
        CenterlineRenderer(vtkSmartPointer<vtkRenderer> renderer);

        void Update(const Fiber* newCenterline);

        void KeyPressed(const std::basic_string<char>& value) override;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_RENDERER_H
