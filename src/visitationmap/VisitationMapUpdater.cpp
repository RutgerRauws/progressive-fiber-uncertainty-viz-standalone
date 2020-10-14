//
// Created by rutger on 10/1/20.
//

#include "VisitationMapUpdater.h"

#include <utility>
#include <sstream>
#include <cmath>
#include <iostream>
#include <cstring>
#include <fstream>

VisitationMapUpdater::VisitationMapUpdater(double* bounds, double spacing)
    : xmin(bounds[0]), xmax(bounds[1]),
      ymin(bounds[2]), ymax(bounds[3]),
      zmin(bounds[4]), zmax(bounds[5]),
      spacing(spacing)
{
    //TODO: Look into fixing double to int conversion.
    width =  std::ceil( std::abs(xmin - xmax) / spacing);
    height = std::ceil(std::abs(ymin - ymax) / spacing);
    depth =  std::ceil(std::abs(zmin - zmax) / spacing);

    initialize();
}

void VisitationMapUpdater::initialize()
{
    /***
     *
     * Setting up OpenGL
     *
     */
    GLenum err = glewInit();
    if(err != GLEW_OK)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        throw std::runtime_error(reinterpret_cast<const char *>(glewGetErrorString(err)));
    }

    std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

    if(!GLEW_VERSION_4_3)
    {
        throw std::runtime_error("OpenGL version 4.3 is not supported.");
    }

    if(!GLEW_ARB_shader_storage_buffer_object)
    {
        /* Problem: we cannot use SSBOs, which is necessary to keep our algorithm performant. */
        throw std::runtime_error("SSBOs are not supported for this graphics card (missing ARB_shader_storage_buffer_object).");
    }

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    std::string source = readStringFromFile(COMPUTE_SHADER_PATH);
    const char* source_ptr = source.c_str();
    glShaderSource(shader,
                   1,
                   &source_ptr,
                   NULL
    );
    glCompileShader(shader);

    checkForErrors(shader); //Check for compilation errors

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    checkForErrors(shader); //Check for linking errors and validate program

    glUseProgram(program);

    /***
     *
     * Preparing inputs
     *
     */
    //Visitation Map Properties
    GLint vmProp_loc;
    vmProp_loc = glGetUniformLocation(program, "vmp.xmin");
    glProgramUniform1d(program, vmProp_loc, xmin);
    vmProp_loc = glGetUniformLocation(program, "vmp.xmax");
    glProgramUniform1d(program, vmProp_loc, xmax);
    vmProp_loc = glGetUniformLocation(program, "vmp.ymin");
    glProgramUniform1d(program, vmProp_loc, ymin);
    vmProp_loc = glGetUniformLocation(program, "vmp.ymax");
    glProgramUniform1d(program, vmProp_loc, ymax);
    vmProp_loc = glGetUniformLocation(program, "vmp.zmin");
    glProgramUniform1d(program, vmProp_loc, zmin);
    vmProp_loc = glGetUniformLocation(program, "vmp.zmax");
    glProgramUniform1d(program, vmProp_loc, zmax);

    vmProp_loc = glGetUniformLocation(program, "vmp.cellSize");
    glProgramUniform1d(program, vmProp_loc, spacing);

    vmProp_loc = glGetUniformLocation(program, "vmp.width");
    glProgramUniform1ui(program, vmProp_loc, width);
    vmProp_loc = glGetUniformLocation(program, "vmp.height");
    glProgramUniform1ui(program, vmProp_loc, height);
    vmProp_loc = glGetUniformLocation(program, "vmp.depth");
    glProgramUniform1ui(program, vmProp_loc, depth);

    //Visitation Map frequencies itself
    unsigned int frequency_data[width * height * depth];
    std::fill_n(frequency_data, width * height * depth, 0);

    GLuint frequency_map_ssbo;
    glGenBuffers(1, &frequency_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int) * width * height * depth, &frequency_data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo);
//    glDispatchCompute(1, 1, 1); //(nr-of-segments / local-group-size, 1, 1)
    glDispatchCompute(10, 1, 1);

    //Sync here to make writes visible
    //glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    //glMemoryBarrier(GL_COMPUTE_SHADER_BIT);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /***
     *
     * Preparing outputs
     *
     */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo);
    GLuint* ptr = (GLuint*) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    std::memcpy(frequency_data, ptr, sizeof(unsigned int) * width * height * depth);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    unsigned int count = 0;
    for(unsigned int i = 0; i < width * height * depth; i++)
    {
        if(frequency_data[i] > 1) {
            std::cout << frequency_data[i] << std::endl;
            count++;
        }
    }

//    std::cout << "Counted " << count << " items of the " << width * height * depth << " total." << std::endl;
}


std::string VisitationMapUpdater::readStringFromFile(const std::string& path)
{
    std::ifstream t(path);
    std::stringstream buffer;
    buffer << t.rdbuf();

    return buffer.str();
}

void VisitationMapUpdater::checkForErrors(GLuint shader)
{
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if(success == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        GLchar errorLog[maxLength];
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

        // Provide the infolog in whatever manor you deem best.
        // Exit with failure.
        glDeleteShader(shader); // Don't leak the shader.

        throw std::runtime_error(std::string("Shader failed to compile/link:\n") + std::string(errorLog));
    }
}
