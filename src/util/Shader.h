//
// Created by rutger on 10/7/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADER_H

#include <GL/glu.h>
#include <stdexcept>

class ShaderError: public std::runtime_error
{
public:
    ShaderError(const std::string& message): std::runtime_error(message) {};
};

class Shader
{
    GLuint id;

public:
    Shader(const std::string& source, GLenum type);
    ~Shader();

    void Compile();
    GLuint GetId();

    static Shader* LoadFromFile(const std::string& path, GLenum type);
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADER_H
