//
// Created by rutger on 11/16/20.
//
#include <libs/glm/ext.hpp>
#include <ostream>
#include <QtGui/QOffscreenSurface>
#include <QMouseEvent>
#include <QtGui/QOpenGLDebugLogger>
#include "src/visitationmap/VisitationMap.h"
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

OGLWidget::OGLWidget(QWidget* parent)
        : QOpenGLWidget(parent),
          mouseDelta(glm::ivec2(0, 0))
{
    format.setVersion(4, 3);
    format.setProfile( QSurfaceFormat::CoreProfile );
    #ifdef DEBUG
    format.setOption(QSurfaceFormat::DebugContext);
    #endif
    format.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(format);
    this->setFormat(format);

    create();
    makeCurrent();

    pTimer = new QTimer(this);
    connect(pTimer, &QTimer::timeout, this, &OGLWidget::render);
}

OGLWidget::~OGLWidget()
{
    delete pTimer;
}

void OGLWidget::onMessageLogged( QOpenGLDebugMessage message )
{
    qDebug() << message;
}

void OGLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    #ifdef DEBUG
        QOpenGLContext *ctx = QOpenGLContext::currentContext();
        logger = new QOpenGLDebugLogger(this);

        connect(logger, SIGNAL(messageLogged(QOpenGLDebugMessage)), this, SLOT(onMessageLogged(QOpenGLDebugMessage)), Qt::DirectConnection);

        logger->initialize(); // initializes in the current context, i.e. ctx

        logger->startLogging();
        logger->enableMessages();
    #endif

    makeCurrent();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pTimer->start(RENDER_INTERVAL_MS);
}

void OGLWidget::paintGL()
{
    makeCurrent();

    // clear the window with black color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    camera.cameraFront += glm::vec3(1, 0, 0);
//    camera.viewMatrix = glm::lookAt(camera.cameraPos,
//                                         camera.cameraPos + camera.cameraFront,
//                                         camera.cameraUp);

    if(initialized)
    {
        visitationMapUpdater->Update();
        centerlineRenderer->Render();
        fiberRenderer->Render();
        visitationMapRenderer->Render();
    }
}

void OGLWidget::resizeGL(int w, int h)
{
    if(initialized)
    {
        camera->ResizeWindow(w, h);
    }
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
    movementHandler->MouseScroll(event->angleDelta().y());
}

