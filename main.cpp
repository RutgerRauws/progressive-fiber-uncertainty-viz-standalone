#include "main.h"

#include <iostream>
#include <X11/Xlib.h>

#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCallbackCommand.h>

#include "KeyPressInteractorStyle.h"
#include "FiberPublisher.h"
#include "FiberRenderer.h"
#include "VisitationMap.h"
#include "VisitationMapUpdater.h"
#include "VisitationMapRenderer.h"
#include "CenterlineRenderer.h"

int main()
{
    XInitThreads();

    std::cout << "Application started." << std::endl;

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

    FiberPublisher fiberPublisher(INPUT_FILE_NAME);
//    FiberPublisher fiberPublisher(INPUT_FILE_NAMES);

    VisitationMap visitationMap(CELL_SIZE);
    VisitationMap visitationMapSplatted(0.25f);
    VisitationMapUpdater visitationMapUpdater(visitationMap, visitationMapSplatted, SPLAT_KERNEL_RADIUS);

    CenterlineRenderer centerlineRenderer(renderer);
    FiberRenderer fiberRenderer(renderer);

    //VisitationMapDebugRenderer visitationMapDebugRenderer(visitationMap, renderer);
//    VisitationMapRenderer visitationMapRenderer(visitationMap, renderer);
    VisitationMapRenderer visitationMapRenderer(visitationMapSplatted, renderer);
    keypressHandler->AddObserver("u", &visitationMapRenderer); //Increasing isovalue
    keypressHandler->AddObserver("j", &visitationMapRenderer); //Decreasing isovalue
    keypressHandler->AddObserver("s", &visitationMapRenderer); //Toggle hull smoothing
    keypressHandler->AddObserver("f", &fiberRenderer); //Toggle rendering of fibers.
    keypressHandler->AddObserver("p", &fiberRenderer); //Toggle rendering of points of fibers.
    keypressHandler->AddObserver("c", &centerlineRenderer); //Toggle rendering of centerline.

    fiberPublisher.RegisterObserver(fiberRenderer);
    fiberPublisher.RegisterObserver(centerlineRenderer);
    fiberPublisher.RegisterObserver(visitationMapUpdater);
    fiberPublisher.RegisterObserver(visitationMapRenderer);
    fiberPublisher.Start();

    //renderWindowInteractor->Initialize();
    renderWindowInteractor->Start();

    //When we reach this point, the renderWindowInteractor has been terminated by the KeyPressInteractorStyle
    fiberPublisher.Stop();

    return EXIT_SUCCESS;
}

void render_callback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData)
{
    auto *renderWindowInteractor = static_cast<vtkRenderWindowInteractor*>(caller);
    renderWindowInteractor->Render();
}