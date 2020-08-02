//
// Created by rutger on 7/2/20.
//

#include <functional>
#include <thread>
#include <utility>
#include "FiberPublisher.h"

FiberPublisher::FiberPublisher(vtkSmartPointer<vtkPolyData> fiberPolyData)
    : keepAddingFibers(true), fiberPolyData(std::move(fiberPolyData))
{}

FiberPublisher::~FiberPublisher()
{
    Stop();

    for(Fiber* fiber : fibers)
    {
        delete fiber;
    }
}

void FiberPublisher::publishFibers_t()
{
    std::cout << "Started fiber publisher thread!" << std::endl;
    
    fiberPolyData->GetLines()->InitTraversal();
    vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();

    while(fiberPolyData->GetLines()->GetNextCell(idList) && keepAddingFibers)
    {
        auto* fiber = new Fiber();
        fibers.emplace_back(fiber);
        
        for(vtkIdType id = 0; id < idList->GetNumberOfIds(); id++)
        {
            double point[3];
            fiberPolyData->GetPoint(idList->GetId(id), point);

            fiber->AddPoint(point[0], point[1], point[2]);
        }

        for(FiberObserver& o : observers)
        {
            o.NewFiber(fiber);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(PROGRESSIVE_INTERVAL_MS));
    }

    std::cout << "Finished fiber publisher thread!" << std::endl;
}

void FiberPublisher::Start()
{
    if(!keepAddingFibers)
    {
        throw std::runtime_error("Cannot start fiber publisher again after it has been stopped.");
    }
    keepAddingFibers = true;
    publishThread = std::thread(&FiberPublisher::publishFibers_t, this);
}

void FiberPublisher::Stop()
{
    if(!keepAddingFibers) { return; }
    keepAddingFibers = false;
    publishThread.join();
}

void FiberPublisher::RegisterObserver(FiberObserver& o)
{
    observers.emplace_back(o);
}