//
// Created by rutger on 11/17/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRIANGLERENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRIANGLERENDERER_H

#include "GL.h"

class TriangleRenderer
{
private:
    GL& gl;

    unsigned int shaderProgram;
    unsigned int VBO, VAO;

    float vertices[9] = {
            -0.5f, -0.5f, 0.0f,
            0.5f, -0.5f, 0.0f,
            0.0f,  0.5f, 0.0f
    };

public:
    TriangleRenderer(GL& gl);

    void Render();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRIANGLERENDERER_H
