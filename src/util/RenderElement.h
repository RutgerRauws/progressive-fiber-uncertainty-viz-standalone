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
        Shader* geometryShader = nullptr;
        Shader* fragmentShader = nullptr;
        ShaderProgram* shaderProgram = nullptr;

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

            if(!geometryShaderPath.empty())
            {
                try
                {
                    geometryShader = Shader::LoadFromFile(geometryShaderPath, GL_GEOMETRY_SHADER);
                    geometryShader->Compile();
                }
                catch(const ShaderError& e)
                {
                    std::cerr << "Could not compile geometry shader: " << e.what() << std::endl;
                    throw e;
                }
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


            if(geometryShaderPath.empty())
            {
                Shader *shaders[2] = {vertexShader, fragmentShader};
                shaderProgram = new ShaderProgram(shaders, 2);
            }
            else
            {
                Shader *shaders[3] = {vertexShader, geometryShader, fragmentShader};
                shaderProgram = new ShaderProgram(shaders, 3);
            }

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
