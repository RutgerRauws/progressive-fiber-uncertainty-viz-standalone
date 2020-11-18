//
// Created by rutger on 11/17/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRIANGLERENDERER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRIANGLERENDERER_H


class TriangleRenderer
{
private:
    unsigned int shaderProgram;
    unsigned int VBO, VAO;

    float vertices[9] = {
            -0.5f, -0.5f, 0.0f,
            0.5f, -0.5f, 0.0f,
            0.0f,  0.5f, 0.0f
    };

public:
    TriangleRenderer();

    void Render();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_TRIANGLERENDERER_H
