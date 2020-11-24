//
// Created by rutger on 11/16/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_OGL_WIDGET_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_OGL_WIDGET_H


#include <QtWidgets/QOpenGLWidget>
#include <QtCore/QTimer>
#include <QOpenGLDebugLogger>

#include "src/visitationmap/VisitationMapUpdater.h"
#include "src/visitationmap/VisitationMapRenderer.h"
#include "src/centerline/CenterlineRenderer.h"
#include "src/util/FiberRenderer.h"
#include "src/util/FiberPublisher.h"
#include "src/util/TriangleRenderer.h"
#include "src/dwi/DWIRenderer.h"

class OGLWidget : public QOpenGLWidget, public QOpenGLFunctions_4_3_Core
{
    Q_OBJECT

    private:
        const unsigned int RENDER_INTERVAL_MS = 33; //30fps

        QSurfaceFormat format;
        QOpenGLDebugLogger* logger = nullptr;

        bool initialized = false;
        VisitationMapUpdater* visitationMapUpdater = nullptr;
        Camera* camera = nullptr;
        MovementHandler* movementHandler = nullptr;

        std::vector<std::reference_wrapper<RenderElement>> renderers;

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
        void onMessageLogged(QOpenGLDebugMessage message);

    public:
        explicit OGLWidget(QWidget *parent = 0);
        ~OGLWidget();

        void SetInput(
            VisitationMapUpdater* visitationMapUpdater,
            Camera* camera,
            MovementHandler* movementHandler
        )
        {
            this->visitationMapUpdater = visitationMapUpdater;
            this->camera = camera;
            this->movementHandler = movementHandler;
            this->initialized = true;
            resizeGL(this->width(), this->height());
        };

        void AddRenderer(RenderElement& renderer)
        {
            renderers.emplace_back(renderer);
        }
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_OGL_WIDGET_H
