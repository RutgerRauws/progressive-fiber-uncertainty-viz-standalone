#version 430

//
// Uniforms
//
uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

//
// In- and outputs
//
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
  gl_Position = projMat * viewMat * modelMat * vec4(position, 1.0);
  TexCoord = aTexCoord;
}