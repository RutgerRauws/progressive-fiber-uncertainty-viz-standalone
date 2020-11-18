//
// Created by rutger on 11/16/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_OGL_WIDGET_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_OGL_WIDGET_H


#include "src/visitationmap/VisitationMapUpdater.h"
#include "src/visitationmap/VisitationMapRenderer.h"
#include "src/centerline/CenterlineRenderer.h"
#include "src/util/FiberRenderer.h"
#include "src/util/FiberPublisher.h"
#include <QtWidgets/QOpenGLWidget>
#include <QtCore/QTimer>
#include <src/util/TriangleRenderer.h>

class OGLWidget : public QOpenGLWidget
{
    private:
        const unsigned int RENDER_INTERVAL_MS = 33; //30fps

        bool initialized = false;
        VisitationMapUpdater* visitationMapUpdater = nullptr;
        VisitationMapRenderer* visitationMapRenderer = nullptr;
        FiberRenderer* fiberRenderer = nullptr;
        CenterlineRenderer* centerlineRenderer = nullptr;
        Camera* camera = nullptr;
        MovementHandler* movementHandler = nullptr;

        glm::ivec2 lastPosition;
        glm::ivec2 mouseDelta;
        QTimer* pTimer = nullptr;

    protected:
        void initializeGL() override;
        void resizeGL(int w, int h) override;
        void paintGL() override;

        void mousePressEvent(QMouseEvent *event) override;
        void mouseMoveEvent(QMouseEvent *event) override;
        void wheelEvent(QWheelEvent *event) override;

    protected slots:
        void render() { paintGL(); update(); };

    public:
        explicit OGLWidget(QWidget *parent = 0);
        ~OGLWidget();

        void SetInput(
            VisitationMapUpdater* visitationMapUpdater,
            VisitationMapRenderer* visitationMapRenderer,
            FiberRenderer* fiberRenderer,
            CenterlineRenderer* centerlineRenderer,
            Camera* camera,
            MovementHandler* movementHandler
        )
        {
            this->visitationMapUpdater = visitationMapUpdater;
            this->visitationMapRenderer = visitationMapRenderer;
            this->fiberRenderer = fiberRenderer;
            this->centerlineRenderer = centerlineRenderer;
            this->camera = camera;
            this->movementHandler = movementHandler;
            this->initialized = true;
            resizeGL(this->width(), this->height());
        };
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_OGL_WIDGET_H
