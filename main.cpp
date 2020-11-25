#include "main.h"

#include <QtOpenGL/qgl.h>

#include "src/dwi/DWIRenderer.h"
#include "src/dwi/DWIDataReader.h"
#include "src/gui/UserInterface.h"
#include "src/util/Camera.h"
#include "src/interaction/TrackBallMovementHandler.h"
#include "src/interaction/InteractionManager.h"
#include "src/gui/OGLWidget.h"


int main(int argc, char* argv[])
{
    QApplication a(argc, argv);

    UserInterface ui;
    ui.Show();

    QOpenGLContext& context = *dynamic_cast<QOpenGLWidget*>(ui.GetOpenGLWidget())->context();
    GL gl(context);

    FiberPublisher fiberPublisher(INPUT_FILE_NAMES);
    DistanceTablesUpdater distanceTablesUpdater(gl, fiberPublisher.GetNumberOfSeedPoints());

    VisitationMap visitationMap(VisitationMap::CreateTest(gl));
    RegionsOfInterest regionsOfInterest(gl, fiberPublisher.GetNumberOfSeedPoints());
    VisitationMapUpdater visitationMapUpdater(gl, visitationMap, regionsOfInterest, distanceTablesUpdater.GetDistanceTables());

    Camera camera;
    TrackBallMovementHandler movementHandler(camera);

    VisitationMapRenderer visitationMapRenderer(gl, visitationMap, regionsOfInterest, distanceTablesUpdater.GetDistanceTables(), camera);
    CenterlineRenderer centerlineRenderer(gl, distanceTablesUpdater.GetDistanceTables(), camera);
    FiberRenderer fiberRenderer(gl, camera);

    DWIDataReader dwiDataReader("/home/rutger/Desktop/Graduation/standalone/progressive-fiber-uncertainty-viz/data/test-data-dwi-volume.nhdr");
    DWISlice coronalSlice = dwiDataReader.GetCoronalPlane();
    DWIRenderer coronalDWIRenderer(gl, camera, coronalSlice);

    DWISlice axialSlice = dwiDataReader.GetAxialPlane();
    DWIRenderer axialDWIRenderer(gl, camera, axialSlice);

    DWISlice sagittalSlice = dwiDataReader.GetSagittalPlane();
    DWIRenderer sagittalDWIRenderer(gl, camera, sagittalSlice);

    fiberPublisher.RegisterObserver(distanceTablesUpdater);
    fiberPublisher.RegisterObserver(visitationMapUpdater);
    fiberPublisher.RegisterObserver(visitationMapRenderer);
    fiberPublisher.RegisterObserver(fiberRenderer);
    fiberPublisher.RegisterObserver(centerlineRenderer);

    OGLWidget* widget = ui.GetOpenGLWidget();
//    widget->SetInput(&visitationMapUpdater, &visitationMapRenderer, &fiberRenderer, &centerlineRenderer, &camera, &movementHandler, &dwiRenderer);
    widget->SetInput(&visitationMapUpdater, &camera, &movementHandler);
    widget->AddRenderer(fiberRenderer);
    widget->AddRenderer(centerlineRenderer);
    widget->AddRenderer(visitationMapRenderer);
    widget->AddRenderer(coronalDWIRenderer);
    widget->AddRenderer(axialDWIRenderer);
    widget->AddRenderer(sagittalDWIRenderer);

    fiberPublisher.Start();

    a.exec();

    fiberPublisher.Stop();

    return 0;
}