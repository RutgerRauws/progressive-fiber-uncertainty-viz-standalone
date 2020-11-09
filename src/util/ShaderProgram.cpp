#include <stdexcept>
#include <GL/glew.h>
#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(Shader** shaders, GLuint num)
{
    id = glCreateProgram();

    for (GLuint i = 0; i < num; i++)
    {
        glAttachShader(id, shaders[i]->GetId());
    }

    glLinkProgram(id);

    GLint log_length;
    glGetProgramiv(id, GL_INFO_LOG_LENGTH, &log_length);

    if (log_length > 1)
    {
        char* error_log = new char[log_length];
        glGetProgramInfoLog(id, log_length, NULL, error_log);
        throw ShaderError(error_log);
    }
}

ShaderProgram::~ShaderProgram ()
{
    glDeleteProgram(id);
}

void ShaderProgram::Use()
{
    glUseProgram(id);
}

GLuint ShaderProgram::GetId ()
{
    return id;
}