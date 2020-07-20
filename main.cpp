#include <iostream>
#include <chrono>
#include <thread>

#include <vtkSmartPointer.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkPolyDataCollection.h>

#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkLine.h>
#include <X11/Xlib.h>
#include <vtkConeSource.h>

#include "KeyPressInteractorStyle.h"
#include "FiberPublisher.h"
#include "FiberRenderer.h"

const std::string INPUT_FILE_NAME = "./data/FiberBundle_1_Output Volume-label.vtk"; //temporary hardcoded input file

bool KeepAddingFibers = true;

vtkSmartPointer<vtkPolyData> fiberPolyData;
vtkSmartPointer<vtkRenderWindow> renderWindow;

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName);

int main()
{
    XInitThreads();
    
    std::cout << "Application started." << std::endl;
    
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

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(coneActor);
    renderer->SetBackground(0, 0, 0);
    
    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(800, 600);
    renderWindow->AddRenderer(renderer);
    
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);
    
    vtkSmartPointer<KeyPressInteractorStyle> style = vtkSmartPointer<KeyPressInteractorStyle>::New();
    renderWindowInteractor->SetInteractorStyle(style);
    
    renderer->ResetCamera();
    renderWindow->Render();
    renderWindowInteractor->Initialize();
    
    FiberRenderer fiberRenderer(renderer, renderWindow);
    
    FiberPublisher fiberPublisher(fiberPolyData);
    fiberPublisher.RegisterObserver(fiberRenderer);
    
    fiberPublisher.Start();
    
    renderWindowInteractor->Start();
    
    //When we reach this point, the renderWindowInteractor has been terminated by the KeyPressInteractorStyle
    renderWindow->Finalize();
    
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