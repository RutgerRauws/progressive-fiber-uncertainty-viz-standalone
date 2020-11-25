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
in vec3 position;

void main()
{
  gl_Position = vec4(position, 1.0);// projMat * viewMat * modelMat * vec4(position, 1.0);
}