//
// Created by rutger on 10/8/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_RENDER_ELEMENT_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_RENDER_ELEMENT_H

#include <GL/gl.h>
#include <iostream>
#include "glm/mat4x4.hpp"
#include "ShaderProgram.h"
#include "../interaction/MovementHandler.h"

class RenderElement
{
    private:
        virtual void initialize() = 0;

    protected:
        float* vertices;

        //Shader variables
        Shader* vertexShader = nullptr;
        Shader* fragmentShader = nullptr;
        ShaderProgram* shaderProgram = nullptr;

        GLuint vao;
        GLuint vbo;

        const CameraState& cameraState;
        int modelMatLoc = -1;
        int viewMatLoc = -1;
        int projMatLoc = -1;

public:
        RenderElement(const std::string& vertexShaderPath,
                      const std::string& fragmentShaderPath,
                      const CameraState& cameraState
        )
            : cameraState(cameraState),
              vertices(nullptr),
              vao(), vbo()
        {
            //Compile shaders and initialize shader program
            try
            {
                vertexShader = Shader::LoadFromFile(vertexShaderPath, GL_VERTEX_SHADER);
                vertexShader->Compile();
            }
            catch(const ShaderError& e)
            {
                std::cerr << "Could not compile vertex shader: " << e.what() << std::endl;
                throw e;
            }

            try
            {
                fragmentShader = Shader::LoadFromFile(fragmentShaderPath, GL_FRAGMENT_SHADER);
                fragmentShader->Compile();
            }
            catch(const ShaderError& e)
            {
                std::cerr << "Could not compile fragment shader: " << e.what() << std::endl;
                throw e;
            }

            Shader* shaders[2] = {vertexShader, fragmentShader};
            shaderProgram = new ShaderProgram(shaders, 2);

            //Setup buffers
            vertices = nullptr;
        }

        ~RenderElement()
        {
            delete shaderProgram;

            delete vertexShader;
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
