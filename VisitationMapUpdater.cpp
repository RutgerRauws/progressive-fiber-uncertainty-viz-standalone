//
// Created by rutger on 7/23/20.
//

#include "VisitationMapUpdater.h"

#include <utility>
#include "Voxel.h"

VisitationMapUpdater::VisitationMapUpdater(VisitationMap& visitationMap)
        : visitationMap(visitationMap)
{}

void VisitationMapUpdater::NewFiber(Fiber* fiber)
{
    //TODO: This should be based on edges
    for(const Point& point : fiber->GetPoints())
    {
        unsigned int value = visitationMap.FindCell(point);

//        if(value == nullptr)
//        {
//            std::cerr << "No corresponding voxel found." << std::endl;
//            continue;
//        }

        visitationMap.SetCell(point, value + 1);
    }
}