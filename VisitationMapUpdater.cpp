//
// Created by rutger on 7/23/20.
//

#include "VisitationMapUpdater.h"
#include "Cell.h"

VisitationMapUpdater::VisitationMapUpdater(VisitationMap& visitationMap,
                                           VisitationMap& visitationMapSplatted,
                                           double splatKernelRadius)
        : visitationMap(visitationMap),
          visitationMapSplatted(visitationMapSplatted),
          splatKernelRadius(splatKernelRadius)
{}

void VisitationMapUpdater::NewFiber(Fiber* fiber)
{
    //TODO: This should be based on edges
    for(const Point& point : fiber->GetPoints())
    {
        visitationMap.InsertPoint(point);
//        splat(point, splatKernelRadius, fiber);
    }
}

/***
 * Splats a spherical kernel with a given radius around a point.
 *
 * @param point  The point around which the spherical splat should take place.
 * @param radius The radius of the sphere.
 * @param fiber  Which fiber should be added to the cells that are considered part of the sphere.
 */
//void VisitationMapUpdater::splat(const Point& point, double radius, Fiber* fiber)
//{
//    //It's not the prettiest code, but this is mainly to make it as efficient as possible on the CPU.
//    unsigned int x_index = 0;
//    unsigned int y_index = 0;
//    unsigned int z_index = 0;
//
//    visitationMapSplatted.GetIndex(point, &x_index, &y_index, &z_index);
//    Cell* centerCell = visitationMapSplatted.GetCell(x_index, y_index, z_index);
//    centerCell->InsertFiber(*fiber);
//
//    //TODO: is this correct?
//    int radiusInNumberCells = std::ceil(radius / visitationMapSplatted.GetCellSize());
//
//    for(int x_index_offset = -radiusInNumberCells; x_index_offset < radiusInNumberCells; x_index_offset++)
//    {
//        for(int y_index_offset = -radiusInNumberCells; y_index_offset < radiusInNumberCells; y_index_offset++)
//        {
//            for(int z_index_offset = -radiusInNumberCells; z_index_offset < radiusInNumberCells; z_index_offset++)
//            {
//                Cell* cell = visitationMapSplatted.GetCell(
//                        x_index + x_index_offset,
//                        y_index + y_index_offset,
//                        z_index + z_index_offset
//                );
//
//                if(cell == nullptr) { continue; }
//
//                if(isCellInsideSphere(point, radius, cell->GetPosition(), visitationMapSplatted.GetCellSize())
//                   && !cell->Contains(*fiber))
//                {
//                    cell->InsertFiber(*fiber);
//                }
//            }
//        }
//    }
//}

bool VisitationMapUpdater::isCellInsideSphere(const Point& center, double radius, const Point& point, double cellSize)
{
    double halfSize = cellSize / 2.0f;

    if(isPointInsideSphere(center, radius, point.X - halfSize, point.Y - halfSize, point.Z - halfSize)
    || isPointInsideSphere(center, radius, point.X - halfSize, point.Y - halfSize, point.Z + halfSize)
    || isPointInsideSphere(center, radius, point.X - halfSize, point.Y + halfSize, point.Z - halfSize)
    || isPointInsideSphere(center, radius, point.X - halfSize, point.Y + halfSize, point.Z + halfSize)
    || isPointInsideSphere(center, radius, point.X + halfSize, point.Y - halfSize, point.Z - halfSize)
    || isPointInsideSphere(center, radius, point.X + halfSize, point.Y - halfSize, point.Z + halfSize)
    || isPointInsideSphere(center, radius, point.X + halfSize, point.Y + halfSize, point.Z - halfSize)
    || isPointInsideSphere(center, radius, point.X + halfSize, point.Y + halfSize, point.Z + halfSize)
    )
    {
        return true;
    }
    else
    {
        return false;
    }
}


bool VisitationMapUpdater::isPointInsideSphere(const Point& center, double radius, double x, double y, double z)
{
    return pow(x - center.X, 2)
        +  pow(y - center.Y, 2)
        +  pow(z - center.Z, 2) <= std::pow(radius, 2);

//    return pow(point.X - center.X, 2)
//        +  pow(point.Y - center.Y, 2)
//        +  pow(point.Z - center.Z, 2) <= std::pow(radius, 2);
}
