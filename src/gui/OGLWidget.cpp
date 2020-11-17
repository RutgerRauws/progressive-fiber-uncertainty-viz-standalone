//
// Created by rutger on 11/16/20.
//

#include <GL/glew.h>

#include <libs/glm/ext.hpp>
#include "src/centerline/DistanceTablesUpdater.h"
#include <ostream>
#include <QtGui/QOffscreenSurface>
#include <QObject>
#include <QMouseEvent>
#include "src/visitationmap/VisitationMap.h"

#include "src/util/FiberRenderer.h"
#include "src/centerline/CenterlineRenderer.h"
#include "src/util/FiberPublisher.h"
#include "main.h"
#include "OGLWidget.h"

void GLAPIENTRY
MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
    fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
             ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
             type, severity, message );
}

OGLWidget::OGLWidget(QWidget *parent)
        : QOpenGLWidget(parent),
          mouseDelta(glm::ivec2(0, 0))
{
    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile( QSurfaceFormat::CoreProfile );
    format.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(format);
    this->setFormat(format);

    create();
    makeCurrent();

    modelMatrix = glm::mat4(1.0f);
    viewMatrix = glm::mat4(1.0f);
    viewMatrix = glm::translate(viewMatrix, glm::vec3(0, 0, -3));
    projectionMatrix = glm::mat4(0); //It's set by resizeGL() anyways

    movementHandler = new MovementHandler(modelMatrix, viewMatrix, projectionMatrix);

    pTimer = new QTimer(this);
    connect(pTimer, &QTimer::timeout, this, &OGLWidget::render);
}

OGLWidget::~OGLWidget()
{
    delete pTimer;

    delete fiberPublisher;
    delete distanceTablesUpdater;

    delete visitationMap;
    delete regionsOfInterest;
    delete visitationMapUpdater;

    delete visitationMapRenderer;
    delete centerlineRenderer;
    delete fiberRenderer;

    delete movementHandler;
}


void OGLWidget::initializeGL()
{
    glewExperimental = GL_TRUE;

    GLenum err = glewInit();
    if(err != GLEW_OK)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        throw std::runtime_error(reinterpret_cast<const char *>(glewGetErrorString(err)));
    }
    std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

    if(!GLEW_VERSION_4_3)
    {
        throw std::runtime_error("OpenGL version 4.3 is not supported.");
    }

    if(!GLEW_ARB_shader_storage_buffer_object)
    {
        /* Problem: we cannot use SSBOs, which is necessary to keep our algorithm performant. */
        throw std::runtime_error("SSBOs are not supported for this graphics card (missing ARB_shader_storage_buffer_object).");
    }

    #ifdef DEBUG
    // During init, enable debug output
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(MessageCallback, 0);
    #endif


    fiberPublisher = new FiberPublisher(INPUT_FILE_NAMES);

    distanceTablesUpdater = new DistanceTablesUpdater (fiberPublisher->GetNumberOfSeedPoints());

    visitationMap = new VisitationMap(VisitationMap::CreateTest());
    regionsOfInterest = new RegionsOfInterest (fiberPublisher->GetNumberOfSeedPoints());
    visitationMapUpdater = new VisitationMapUpdater(*visitationMap, *regionsOfInterest, distanceTablesUpdater->GetDistanceTables());

    visitationMapRenderer = new VisitationMapRenderer(*visitationMap, *regionsOfInterest, distanceTablesUpdater->GetDistanceTables(), movementHandler->GetCameraState());
    fiberRenderer = new FiberRenderer(movementHandler->GetCameraState());
    centerlineRenderer = new CenterlineRenderer(distanceTablesUpdater->GetDistanceTables(), movementHandler->GetCameraState());

    fiberPublisher->RegisterObserver(*distanceTablesUpdater);
    fiberPublisher->RegisterObserver(*visitationMapUpdater);
    fiberPublisher->RegisterObserver(*visitationMapRenderer);
    fiberPublisher->RegisterObserver(*fiberRenderer);
    fiberPublisher->RegisterObserver(*centerlineRenderer);


    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pTimer->start(RENDER_INTERVAL_MS);
    fiberPublisher->Start();
}

void OGLWidget::paintGL()
{
    // clear the window with black color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    cameraState.cameraFront += glm::vec3(1, 0, 0);
//    cameraState.viewMatrix = glm::lookAt(cameraState.cameraPos,
//                                         cameraState.cameraPos + cameraState.cameraFront,
//                                         cameraState.cameraUp);

    //Actual draw calls
    visitationMapUpdater->Update();
    visitationMapRenderer->Render();
    centerlineRenderer->Render();
    fiberRenderer->Render();
}

void OGLWidget::resizeGL(int w, int h)
{
    projectionMatrix = glm::perspective(
            glm::radians(FOV),
            (float)w / (float)h,
            0.1f,
            10000.0f
    );
}

void OGLWidget::mousePressEvent(QMouseEvent *event)
{
    lastPosition.x = event->x();
    lastPosition.y = event->y();
}

void OGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    mouseDelta.x = lastPosition.x - event->x();
    mouseDelta.y = lastPosition.y - event->y();

    lastPosition.x = event->x();
    lastPosition.y = event->y();

    movementHandler->MouseMovement(mouseDelta);
}

void OGLWidget::wheelEvent(QWheelEvent *event)
{
//    QWidget::wheelEvent(event);
}

