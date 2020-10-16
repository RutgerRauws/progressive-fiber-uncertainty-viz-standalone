#include <fstream>
#include <GL/glew.h>
#include <sstream>
#include "Shader.h"

Shader::Shader(const std::string& source, GLenum type)
{
    id = glCreateShader(type);
    const char* source_ptr = source.c_str();
    glShaderSource(id, 1, &source_ptr, NULL);
}

Shader::~Shader ()
{
    glDeleteShader(id);
}

void Shader::Compile()
{
    glCompileShader(id);

    GLint log_length;
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &log_length);

    if (log_length > 1)
    {
        char* error_log = new char[log_length];
        glGetShaderInfoLog(id, log_length, NULL, error_log);

        throw ShaderError(error_log);
    }
}

GLuint Shader::GetId()
{
    return id;
}

Shader* Shader::LoadFromFile(const std::string& path, GLenum type)
{
    std::ifstream t(path);
    std::stringstream buffer;
    buffer << t.rdbuf();

    return new Shader(buffer.str(), type);
}