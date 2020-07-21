#include <iostream>
#include <thread>

#include <vtkSmartPointer.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkPolyDataCollection.h>

#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkLine.h>
#include <X11/Xlib.h>
#include <vtkConeSource.h>

#include "KeyPressInteractorStyle.h"
#include "FiberPublisher.h"
#include "FiberRenderer.h"
#include "VisitationMap.h"
#include "VisitationMapRenderer.h"

const std::string INPUT_FILE_NAME = "./data/FiberBundle_1_Output Volume-label.vtk"; //temporary hardcoded input file

bool KeepAddingFibers = true;

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName);

int main()
{
    XInitThreads();

    std::cout << "Application started." << std::endl;

    vtkSmartPointer<vtkPolyData> fiberPolyData;

    try {
        fiberPolyData = readPolyData(INPUT_FILE_NAME);
        std::cout << "Input has " << fiberPolyData->GetNumberOfLines() << " fibers." << std::endl;
    }
    catch( const std::invalid_argument& e ) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    vtkSmartPointer<vtkConeSource> coneSource = vtkSmartPointer<vtkConeSource>::New();
    vtkSmartPointer<vtkPolyDataMapper> coneMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    coneMapper->SetInputConnection(coneSource->GetOutputPort());
    vtkSmartPointer<vtkActor> coneActor = vtkSmartPointer<vtkActor>::New();
    coneActor->SetMapper(coneMapper);

    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    camera->SetPosition(0, 0, 325);
    camera->SetViewUp(1, 0, 0);
    camera->SetFocalPoint(0, 0, 0);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(coneActor);
    renderer->SetBackground(0, 0, 0);
    renderer->SetActiveCamera(camera);

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1920, 1080);
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkSmartPointer<KeyPressInteractorStyle> style = vtkSmartPointer<KeyPressInteractorStyle>::New();
    renderWindowInteractor->SetInteractorStyle(style);

    renderWindow->Render();
    renderWindowInteractor->Initialize();

    FiberPublisher fiberPublisher(fiberPolyData);

    FiberRenderer fiberRenderer(renderer, renderWindow);
    VisitationMap visitationMap(fiberPolyData->GetBounds());
    VisitationMapRenderer visitationMapRenderer(visitationMap, renderer, renderWindow);

    fiberPublisher.RegisterObserver(fiberRenderer);
    fiberPublisher.RegisterObserver(visitationMapRenderer);
    fiberPublisher.Start();

    renderWindowInteractor->Start();

    //When we reach this point, the renderWindowInteractor has been terminated by the KeyPressInteractorStyle
    fiberPublisher.Stop();

    return EXIT_SUCCESS;
}

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName)
{
    vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();
    
    if(!reader->IsFilePolyData())
    {
        throw std::invalid_argument("The file input is not polygon data");
    }
    
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->CopyStructure(reader->GetPolyDataOutput());
    
    return polyData;
}