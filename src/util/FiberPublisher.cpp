//
// Created by rutger on 7/2/20.
//

#include <functional>
#include <thread>
#include <utility>
#include <vtkGenericDataObjectReader.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include "FiberPublisher.h"

FiberPublisher::FiberPublisher(vtkSmartPointer<vtkPolyData> fiberPolyData)
    : keepAddingFibers(true)
{
    fiberPolyDatas.emplace_back(fiberPolyData);
}

FiberPublisher::FiberPublisher(std::vector<vtkSmartPointer<vtkPolyData>> fiberPolyDatas)
        : keepAddingFibers(true),
          fiberPolyDatas(std::move(fiberPolyDatas))
{}

FiberPublisher::FiberPublisher(const std::string& path)
    : FiberPublisher(FiberPublisher::loadFromFile(path))
{}

FiberPublisher::FiberPublisher(const std::vector<std::string>& paths)
    : FiberPublisher(FiberPublisher::loadFromFiles(paths))
{}


FiberPublisher::~FiberPublisher()
{
    Stop();

    lock.lock();
    for(Fiber* fiber : fibers)
    {
        delete fiber;
    }
    lock.unlock();
}

void FiberPublisher::publishFibers_t(vtkSmartPointer<vtkPolyData> fiberPolyData, unsigned int seedPointId)
{
    std::cout << "Started fiber publisher thread!" << std::endl;

    fiberPolyData->GetLines()->InitTraversal();
    vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();

    while(fiberPolyData->GetLines()->GetNextCell(idList) && keepAddingFibers)
    {
        auto* fiber = new Fiber(seedPointId);

        for(vtkIdType id = 0; id < idList->GetNumberOfIds() - 1; id++)
        {
            double p1[3];
            fiberPolyData->GetPoint(idList->GetId(id), p1);

            double p2[3];
            fiberPolyData->GetPoint(idList->GetId(id + 1), p2);

            fiber->AddSegment(
                glm::vec3(p1[0], p1[1], p1[2]),
                glm::vec3(p2[0], p2[1], p2[2])
            );
        }

        lock.lock();
        fibers.emplace_back(fiber); //Todo: we need a mutex here still
        lock.unlock();

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

    unsigned int maxSeedPointId = 0;

    for(vtkSmartPointer<vtkPolyData> fiberPolyData : fiberPolyDatas)
    {
        unsigned int seedPointId = maxSeedPointId++;
        publishThreads.emplace_back(std::thread(&FiberPublisher::publishFibers_t, this, fiberPolyData, seedPointId));
    }
}

void FiberPublisher::Stop()
{
    if(!keepAddingFibers) { return; }
    keepAddingFibers = false;

    for(std::thread& publishThread : publishThreads)
    {
        publishThread.join();
    }
}

void FiberPublisher::RegisterObserver(FiberObserver& o)
{
    observers.emplace_back(o);
}

vtkSmartPointer<vtkPolyData> FiberPublisher::loadFromFile(const std::string& path)
{
    std::cout << "Loading polygon file... " << std::flush;

    vtkSmartPointer<vtkPolyData> fiberPolyData = vtkSmartPointer<vtkPolyData>::New();

    try
    {
        vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
        reader->SetFileName(path.c_str());
        reader->Update();

        if(!reader->IsFilePolyData())
        {
            throw std::invalid_argument("The file input is not polygon data");
        }

        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
        polyData->CopyStructure(reader->GetPolyDataOutput());

        //TODO: Confirm that this transformation below is not necessary.
//        vtkSmartPointer<vtkTransform> rotation = vtkSmartPointer<vtkTransform>::New();
//        rotation->RotateZ(-90);

//        vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
//        transformFilter->SetInputData(polyData);
//        transformFilter->SetTransform(rotation);
//        transformFilter->Update();

//        fiberPolyData->CopyStructure(transformFilter->GetOutput());
        fiberPolyData->CopyStructure(polyData);
        std::cout << "Complete." << std::endl;

        std::cout << "Input has " << fiberPolyData->GetNumberOfLines() << " fibers." << std::endl;
    }
    catch(const std::invalid_argument& e)
    {
        throw std::runtime_error(e.what());
    }

    return fiberPolyData;
}

std::vector<vtkSmartPointer<vtkPolyData>> FiberPublisher::loadFromFiles(const std::vector<std::string>& paths)
{
    std::vector<vtkSmartPointer<vtkPolyData>> _fiberPolyDatas;

    for(const std::string& path : paths)
    {
        _fiberPolyDatas.emplace_back(
            FiberPublisher::loadFromFile(path)
        );
    }

    return _fiberPolyDatas;
}