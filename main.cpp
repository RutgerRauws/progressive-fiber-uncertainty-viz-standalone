#include <iostream>
#include <thread>

#include <vtkSmartPointer.h>
#include <vtkGenericDataObjectReader.h>

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <X11/Xlib.h>
#include <vtkCallbackCommand.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include "KeyPressInteractorStyle.h"
#include "FiberPublisher.h"
#include "FiberRenderer.h"
#include "VisitationMap.h"
#include "VisitationMapUpdater.h"
//#include "VisitationMapDebugRenderer.h"
#include "VisitationMapRenderer.h"
#include "CenterlineRenderer.h"

//temporary hardcoded input file
//const std::string INPUT_FILE_NAME = "./data/corpus-callosum.vtk";
//const std::string INPUT_FILE_NAME = "./data/fiber-samples-without-outliers.vtk";
//const std::string INPUT_FILE_NAME = "./data/fiber-samples-with-outliers.vtk";
//const std::string INPUT_FILE_NAME = "./data/cst-1.vtk";
const std::string INPUT_FILE_NAME = "./data/cst-20.vtk";
//const std::string INPUT_FILE_NAME = "./data/slf-1.vtk";
//const std::string INPUT_FILE_NAME = "./data/slf-20.vtk";
//const std::string INPUT_FILE_NAME = "./data/cc-5.vtk";

const unsigned int RENDER_INTERVAL_MS = 33; //30fps

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName);
void render_callback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData);

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

    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    camera->SetPosition(0, 0, 325);
    camera->SetFocalPoint(0, 0, 0);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(0, 0, 0);
    renderer->SetActiveCamera(camera);

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(1920, 1080);
    renderWindow->SetWindowName("Progressive Fiber Uncertainty Visualization");
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);
    renderWindowInteractor->Initialize();

    /*
     * Adding callbacks
     */
    vtkSmartPointer<vtkCallbackCommand> renderCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    renderCallback->SetCallback(render_callback);
    renderWindowInteractor->AddObserver (vtkCommand::TimerEvent, renderCallback);
    renderWindowInteractor->CreateRepeatingTimer(RENDER_INTERVAL_MS);

    vtkSmartPointer<KeyPressInteractorStyle> keypressHandler = vtkSmartPointer<KeyPressInteractorStyle>::New();
    renderWindowInteractor->SetInteractorStyle(keypressHandler);
    //renderWindowInteractor->AddObserver (vtkCommand::KeyPressEvent, renderCallback);

    /*
     * Starting main functionality
     */
    renderWindow->Render();

    VisitationMap visitationMap(fiberPolyData->GetBounds());
    VisitationMapUpdater visitationMapUpdater(visitationMap);

    FiberPublisher fiberPublisher(fiberPolyData);

    CenterlineRenderer centerlineRenderer(renderer);
    FiberRenderer fiberRenderer(renderer);

    //VisitationMapDebugRenderer visitationMapDebugRenderer(visitationMap, renderer);
    VisitationMapRenderer visitationMapRenderer(visitationMap, renderer);
    keypressHandler->AddObserver("u", &visitationMapRenderer); //Increasing isovalue
    keypressHandler->AddObserver("j", &visitationMapRenderer); //Decreasing isovalue
    keypressHandler->AddObserver("s", &visitationMapRenderer); //Toggle hull smoothing
    keypressHandler->AddObserver("f", &fiberRenderer); //Toggle rendering of fibers.
    keypressHandler->AddObserver("p", &fiberRenderer); //Toggle rendering of points of fibers.
    keypressHandler->AddObserver("c", &centerlineRenderer); //Toggle rendering of centerline.

    fiberPublisher.RegisterObserver(fiberRenderer);
    fiberPublisher.RegisterObserver(centerlineRenderer);
    fiberPublisher.RegisterObserver(visitationMapUpdater);
    fiberPublisher.Start();

    //renderWindowInteractor->Initialize();
    renderWindowInteractor->Start();

    //When we reach this point, the renderWindowInteractor has been terminated by the KeyPressInteractorStyle
    fiberPublisher.Stop();

    return EXIT_SUCCESS;
}

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName)
{
    std::cout << "Loading polygon file... " << std::flush;

    vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();
    
    if(!reader->IsFilePolyData())
    {
        throw std::invalid_argument("The file input is not polygon data");
    }
    
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->CopyStructure(reader->GetPolyDataOutput());

    vtkSmartPointer<vtkTransform> rotation = vtkSmartPointer<vtkTransform>::New();
    rotation->RotateZ(-90);

    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformFilter->SetInputData(polyData);
    transformFilter->SetTransform(rotation);
    transformFilter->Update();

    polyData->CopyStructure(transformFilter->GetOutput());
    std::cout << "Complete." << std::endl;
    return polyData;
}

void render_callback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData)
{
    auto *renderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(caller);
    renderWindowInteractor->Render();
}