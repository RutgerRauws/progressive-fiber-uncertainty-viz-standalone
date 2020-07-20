//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_PUBLISHER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_PUBLISHER_H

#include <vector>
#include <vtkPolyData.h>
#include "FiberObserver.h"

class FiberPublisher
{
private:
    const int PROGRESSIVE_INTERVAL_MS = 100;
    bool keepAddingFibers;
    vtkSmartPointer<vtkPolyData> fiberPolyData;
    std::vector<std::reference_wrapper<FiberObserver>> observers;
    std::thread publishThread;
    
    void publishFibers_t();
    
public:
    explicit FiberPublisher(vtkSmartPointer<vtkPolyData> fiberPolyData);
    ~FiberPublisher();
    
    void Start();
    void Stop();
    void RegisterObserver(FiberObserver& o);
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_PUBLISHER_H
