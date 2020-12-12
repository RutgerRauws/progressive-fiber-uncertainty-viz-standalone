#include "main.h"

#include <QtOpenGL/qgl.h>

#include "src/mri/MRIRenderer.h"
#include "src/mri/MRIDataReader.h"
#include "src/gui/UserInterface.h"
#include "src/util/Camera.h"
#include "src/interaction/TrackBallMovementHandler.h"
#include "src/interaction/InteractionManager.h"
#include "src/gui/OGLWidget.h"
#include "Configuration.h"

int main(int argc, char* argv[])
{
    Configuration& config = Configuration::getInstance();

    QApplication a(argc, argv);

    UserInterface ui;
    ui.Show();

    QOpenGLContext& context = *dynamic_cast<QOpenGLWidget*>(ui.GetOpenGLWidget())->context();
    GL gl(context);

    FiberPublisher fiberPublisher(FiberPublisher::GetVTKFilesInFolder(FIBER_FOLDER));
    DistanceTablesUpdater distanceTablesUpdater(gl, fiberPublisher.GetNumberOfSeedPoints());

    VisitationMap visitationMap(VisitationMap::CreateVisitationMapFromDWIDimensions(gl, DWI_X, DWI_Y, DWI_Z, DWI_SIZE, config.SIDE_SIZE));
    RegionsOfInterest regionsOfInterest(gl, fiberPublisher.GetNumberOfSeedPoints());
    VisitationMapUpdater visitationMapUpdater(gl, visitationMap, regionsOfInterest, distanceTablesUpdater.GetDistanceTables());

    Camera camera;
    TrackBallMovementHandler movementHandler(camera);

    VisitationMapRenderer visitationMapRenderer(gl, visitationMap, regionsOfInterest, distanceTablesUpdater.GetDistanceTables(), camera);
    CenterlineRenderer centerlineRenderer(gl, distanceTablesUpdater.GetDistanceTables(), camera);
    FiberRenderer fiberRenderer(gl, camera);

    MRIDataReader mriDataReader(DWI_PATH);
    MRISlice coronalSlice = mriDataReader.GetCoronalPlane();
    MRIRenderer coronalDWIRenderer(gl, camera, coronalSlice, config.SHOW_CORONAL_PLANE);

    MRISlice axialSlice = mriDataReader.GetAxialPlane();
    MRIRenderer axialDWIRenderer(gl, camera, axialSlice, config.SHOW_AXIAL_PLANE);

    MRISlice sagittalSlice = mriDataReader.GetSagittalPlane();
    MRIRenderer sagittalDWIRenderer(gl, camera, sagittalSlice, config.SHOW_SAGITTAL_PLANE);

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
    widget->AddRenderer(coronalDWIRenderer);
    widget->AddRenderer(axialDWIRenderer);
    widget->AddRenderer(sagittalDWIRenderer);
    widget->AddRenderer(visitationMapRenderer);

    fiberPublisher.Start();

    a.exec();

    fiberPublisher.Stop();

    return 0;
}