//
// Created by rutger on 11/17/20.
//

#include "TriangleRenderer.h"

TriangleRenderer::TriangleRenderer(GL& gl) : gl(gl)
{
    const char *vertexShaderSource = "#version 330 core\n"
                                     "layout (location = 0) in vec3 aPos;\n"
                                     "void main()\n"
                                     "{\n"
                                     "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
                                     "}\0";

    unsigned int vertexShader;
    vertexShader = gl.glCreateShader(GL_VERTEX_SHADER);

    gl.glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    gl.glCompileShader(vertexShader);

    const char *fragmentShaderSource = "#version 330 core\n"
                                       "out vec4 FragColor;\n"
                                       "\n"
                                       "void main()\n"
                                       "{\n"
                                       "    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
                                       "}\0";


    unsigned int fragmentShader;
    fragmentShader = gl.glCreateShader(GL_FRAGMENT_SHADER);
    gl.glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    gl.glCompileShader(fragmentShader);

    shaderProgram = gl.glCreateProgram();
    gl.glAttachShader(shaderProgram, vertexShader);
    gl.glAttachShader(shaderProgram, fragmentShader);
    gl.glLinkProgram(shaderProgram);

    gl.glDeleteShader(vertexShader);
    gl.glDeleteShader(fragmentShader);


    gl.glUseProgram(shaderProgram);

    gl.glGenBuffers(1, &VBO);
    gl.glGenVertexArrays(1, &VAO);

    // 1. bind Vertex Array Object
    gl.glBindVertexArray(VAO);
    // 2. copy our vertices array in a buffer for OpenGL to use
    gl.glBindBuffer(GL_ARRAY_BUFFER, VBO);
    gl.glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // 3. then set our vertex attributes pointers
    gl.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    gl.glEnableVertexAttribArray(0);
}

void TriangleRenderer::Render()
{
    gl.glUseProgram(shaderProgram);
    gl.glBindVertexArray(VAO);

    glDrawArrays(GL_TRIANGLES, 0, 3);
}
