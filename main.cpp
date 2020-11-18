#include "main.h"

#include <src/gui/UserInterface.h>
#include <QtOpenGL/qgl.h>
#include <src/util/Camera.h>
#include "src/interaction/InteractionManager.h"


int main(int argc, char* argv[])
{
    QApplication a(argc, argv);

    UserInterface ui;
    ui.Show();

    FiberPublisher fiberPublisher(INPUT_FILE_NAMES);
    DistanceTablesUpdater distanceTablesUpdater(fiberPublisher.GetNumberOfSeedPoints());

    VisitationMap visitationMap(VisitationMap::CreateTest());
    RegionsOfInterest regionsOfInterest(fiberPublisher.GetNumberOfSeedPoints());
    VisitationMapUpdater visitationMapUpdater(visitationMap, regionsOfInterest, distanceTablesUpdater.GetDistanceTables());

    Camera camera(CAMERA_POS, CAMERA_FRT, CAMERA_UP);

    MovementHandler movementHandler(camera);
    VisitationMapRenderer visitationMapRenderer(visitationMap, regionsOfInterest, distanceTablesUpdater.GetDistanceTables(), camera);
    CenterlineRenderer centerlineRenderer(distanceTablesUpdater.GetDistanceTables(), camera);
    FiberRenderer fiberRenderer(camera);

    fiberPublisher.RegisterObserver(distanceTablesUpdater);
    fiberPublisher.RegisterObserver(visitationMapUpdater);
    fiberPublisher.RegisterObserver(visitationMapRenderer);
    fiberPublisher.RegisterObserver(fiberRenderer);
    fiberPublisher.RegisterObserver(centerlineRenderer);

    OGLWidget* widget = ui.GetOpenGLWidget();
    widget->SetInput(&visitationMapUpdater, &visitationMapRenderer, &fiberRenderer, &centerlineRenderer, &camera, &movementHandler);

    fiberPublisher.Start();

    return a.exec();
}