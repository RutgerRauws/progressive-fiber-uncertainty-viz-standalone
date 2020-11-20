//
// Created by rutger on 11/20/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_GL_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_GL_H

#include <QtGui/QOpenGLFunctions_4_3_Core>

class GL : public QOpenGLFunctions_4_3_Core
{
    public:
        QOpenGLContext& context;

        GL(QOpenGLContext& c) : context(c) { initializeOpenGLFunctions(); };
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_GL_H
