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
        const float FOV = 45.0f;
        const unsigned int RENDER_INTERVAL_MS = 33; //30fps

        FiberPublisher* fiberPublisher = nullptr;
        DistanceTablesUpdater* distanceTablesUpdater = nullptr;

        VisitationMap* visitationMap = nullptr;
        RegionsOfInterest* regionsOfInterest = nullptr;
        VisitationMapUpdater* visitationMapUpdater = nullptr;

        VisitationMapRenderer* visitationMapRenderer = nullptr;
        CenterlineRenderer* centerlineRenderer = nullptr;
        FiberRenderer* fiberRenderer = nullptr;

        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
        glm::mat4 projectionMatrix;

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
        OGLWidget(QWidget *parent = 0);
        ~OGLWidget();

//        void SetInput(
//            FiberPublisher* fiberPublisher,
//            DistanceTablesUpdater* distanceTablesUpdater,
//            VisitationMap* visitationMap,
//            RegionsOfInterest* regionsOfInterest,
//            VisitationMapUpdater* visitationMapUpdater,
//            VisitationMapRenderer* visitationMapRenderer,
//            FiberRenderer* fiberRenderer,
//            CenterlineRenderer* centerlineRenderer
//        );

};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_OGL_WIDGET_H
