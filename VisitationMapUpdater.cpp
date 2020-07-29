//
// Created by rutger on 7/23/20.
//

#include "VisitationMapUpdater.h"

#include <utility>
#include <set>
#include "Voxel.h"

VisitationMapUpdater::VisitationMapUpdater(VisitationMap& visitationMap)
        : visitationMap(visitationMap)
{}

void VisitationMapUpdater::NewFiber(const Fiber &fiber)
{
    std::set<unsigned int> voxelIds;

    //TODO: This should be based on edges
    for(const Point& point : fiber.GetPoints())
    {
        Voxel* voxel = visitationMap.FindCell(point);

        if(voxel == nullptr)
        {
            std::cerr << "No corresponding voxel found." << std::endl;
            continue;
        }

        if(voxelIds.find(voxel->GetID()) != voxelIds.end())
        {
            //Voxel at this position is already incremented
            continue;
        }

        voxel->SetValue(voxel->GetValue() + 1);
        voxelIds.insert(voxel->GetID());
    }
}