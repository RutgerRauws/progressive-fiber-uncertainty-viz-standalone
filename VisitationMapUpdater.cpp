//
// Created by rutger on 7/23/20.
//

#include "VisitationMapUpdater.h"
#include "Cell.h"

VisitationMapUpdater::VisitationMapUpdater(VisitationMap& visitationMap)
        : visitationMap(visitationMap)
{}

void VisitationMapUpdater::NewFiber(Fiber* fiber)
{
    //TODO: This should be based on edges
    for(const Point& point : fiber->GetPoints())
    {
        Cell* cell = visitationMap.FindCell(point);

        if(cell == nullptr)
        {
            std::cerr << "No corresponding voxel found." << std::endl;
            continue;
        }

//        if(!cell->Contains(*fiber))
//        {
        cell->InsertFiber(*fiber);
//        }

//       cell->SetValue(cell->GetValue() + 1);
        //visitationMap.SetCell(point, value + 1);
    }
}