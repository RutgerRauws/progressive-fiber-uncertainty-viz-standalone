//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_RENDER_ELEMENT_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_RENDER_ELEMENT_H

#include <GL/gl.h>

class RenderElement
{
    private:
        virtual void initialize() = 0;

    protected:
        float* vertices;

        GLuint vao;
        GLuint vbo;

    public:
        RenderElement()
            : vertices(nullptr),
              vao(), vbo()
        {}

        virtual void Render() = 0;

        float* GetVertexBufferData()
        {
            return vertices;
        }

        virtual unsigned int GetNumberOfVertices() = 0;
        virtual unsigned int GetNumberOfBytes() = 0;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_RENDER_ELEMENT_H
