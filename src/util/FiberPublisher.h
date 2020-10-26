//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_PUBLISHER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_PUBLISHER_H

#include <vector>
#include <thread>
#include <vtkPolyData.h>
#include <mutex>
#include "FiberObserver.h"

class FiberPublisher
{
private:
    std::mutex lock;

    const int PROGRESSIVE_INTERVAL_MS = 500;
    bool keepAddingFibers;

    std::vector<vtkSmartPointer<vtkPolyData>> fiberPolyDatas;
    std::vector<std::reference_wrapper<FiberObserver>> observers;

    std::vector<Fiber*> fibers;

    std::vector<std::thread> publishThreads;
    
    void publishFibers_t(vtkSmartPointer<vtkPolyData> fiberPolyData, unsigned int seedPointId);
    static vtkSmartPointer<vtkPolyData> loadFromFile(const std::string& path);
    std::vector<vtkSmartPointer<vtkPolyData>> loadFromFiles(const std::vector<std::string>& paths);
    
public:
    explicit FiberPublisher(vtkSmartPointer<vtkPolyData> fiberPolyData);
    explicit FiberPublisher(std::vector<vtkSmartPointer<vtkPolyData>> fiberPolyDatas);
    explicit FiberPublisher(const std::string& path);
    explicit FiberPublisher(const std::vector<std::string>& paths);
    ~FiberPublisher();

    void Start();
    void Stop();
    void RegisterObserver(FiberObserver& o);

    double* GetBounds() const;
    unsigned int GetNumberOfSeedPoints() { return fiberPolyDatas.size(); }
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_PUBLISHER_H
