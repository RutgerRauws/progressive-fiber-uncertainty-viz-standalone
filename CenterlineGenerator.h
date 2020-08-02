//
// Created by rutger on 7/29/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_GENERATOR_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_GENERATOR_H


#include <vtkRenderer.h>
#include <vtkTubeFilter.h>
#include <vtkPolyLine.h>
#include <set>
#include <unordered_set>
#include "FiberObserver.h"
#include "CenterlineRenderer.h"

class CenterlineGenerator : public FiberObserver
{
private:
    static constexpr double DISTANCE_THRESHOLD = 0.1f;

    std::reference_wrapper<CenterlineRenderer> centerlineRenderer;

    std::vector<std::pair<double, Fiber*>> distanceTable;
    const Fiber* centerfiber_ptr;

    static double calculateMinimumDistanceScore(const Fiber& fiber1, const Fiber& fiber2);
    static double calculateMinimumDistanceScore_dm(const Fiber& Fi, const Fiber& Fj);

public:
    explicit CenterlineGenerator(CenterlineRenderer& centerlineRenderer);

    void NewFiber(Fiber* fiber) override;
    const Fiber* GetCenterline() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CENTERLINE_GENERATOR_H
