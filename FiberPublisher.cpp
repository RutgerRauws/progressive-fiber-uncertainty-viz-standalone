//
// Created by rutger on 7/2/20.
//

#include <functional>
#include <thread>
#include "FiberPublisher.h"

FiberPublisher::FiberPublisher(vtkSmartPointer<vtkPolyData> fiberPolyData)
    : keepAddingFibers(true), fiberPolyData(fiberPolyData)
{}

FiberPublisher::~FiberPublisher() { Stop(); }

void FiberPublisher::publishFibers_t()
{
    std::cout << "Started fiber publisher thread!";
    
    fiberPolyData->GetLines()->InitTraversal();
    vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
    
    while(fiberPolyData->GetLines()->GetNextCell(idList) && keepAddingFibers) {
        
        Fiber fiber;
        
        for(vtkIdType id = 0; id < idList->GetNumberOfIds(); id++)
        {
            double* point = fiberPolyData->GetPoint(idList->GetId(id));
            fiber.AddPoint(*point, *(point + 1), *(point + 2));
        }

        for(FiberObserver& o : observers)
        {
            o.NewFiber(fiber);
        }
        
        std::cout << "Published line!" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(PROGRESSIVE_INTERVAL_MS));
    }
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