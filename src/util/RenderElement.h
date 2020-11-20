//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_RENDER_ELEMENT_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_RENDER_ELEMENT_H

#include <GL/gl.h>
#include <iostream>
#include <QtGui/QOpenGLShader>
#include "glm/mat4x4.hpp"
#include "../interaction/MovementHandler.h"

class RenderElement
{
    private:
        virtual void initialize() = 0;

    protected:
        float* vertices;

        //Shader variables
        QOpenGLShader* vertexShader = nullptr;
        QOpenGLShader* geometryShader = nullptr;
        QOpenGLShader* fragmentShader = nullptr;
        QOpenGLShaderProgram* shaderProgram = nullptr;

        GLuint vao;
        GLuint vbo;

        const Camera& camera;
        GLint modelMatLoc = -1;
        GLint viewMatLoc = -1;
        GLint projMatLoc = -1;

public:
        RenderElement(const std::string& vertexShaderPath, const std::string fragmentShaderPath, const Camera& camera)
            : RenderElement(vertexShaderPath, "", fragmentShaderPath, camera)
        {}

        RenderElement(const std::string& vertexShaderPath,
                      const std::string& geometryShaderPath,
                      const std::string& fragmentShaderPath,
                      const Camera& camera
        )
            : camera(camera),
              vertices(nullptr),
              vao(), vbo()
        {
            //Compile shaders and initialize shader program

            vertexShader = new QOpenGLShader(QOpenGLShader::ShaderTypeBit::Vertex); // QOpenGLShader::compileSourceFile(vertexShaderPath);
            vertexShader->compileSourceFile(QString(vertexShaderPath.data()));


            if(!geometryShaderPath.empty())
            {

                geometryShader = new QOpenGLShader(QOpenGLShader::ShaderTypeBit::Geometry);
                geometryShader->compileSourceFile(QString(geometryShaderPath.data()));
            }

            fragmentShader = new QOpenGLShader(QOpenGLShader::ShaderTypeBit::Fragment);
            fragmentShader->compileSourceFile(QString(fragmentShaderPath.data()));

            if(geometryShaderPath.empty())
            {
                shaderProgram = new QOpenGLShaderProgram();
                shaderProgram->addShader(vertexShader);
                shaderProgram->addShader(fragmentShader);
            }
            else
            {
                shaderProgram = new QOpenGLShaderProgram();
                shaderProgram->addShader(vertexShader);
                shaderProgram->addShader(geometryShader);
                shaderProgram->addShader(fragmentShader);
            }

            shaderProgram->link();

            //Setup buffers
            vertices = nullptr;
        }

        ~RenderElement()
        {
            delete shaderProgram;

            delete vertexShader;
            delete geometryShader;
            delete fragmentShader;
        }

        virtual void Render() = 0;

        float* GetVertexBufferData()
        {
            return vertices;
        }

        virtual unsigned int GetNumberOfVertices() = 0;
        virtual unsigned int GetNumberOfBytes() = 0;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_RENDER_ELEMENT_H
