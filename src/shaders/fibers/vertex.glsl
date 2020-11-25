#version 430

//
// Uniforms
//
uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;
//uniform mat4 modelViewProjMat;

//
//Variables
//
vec3 lightPos = vec3(1, 1, 0);
vec3 lightDir = vec3(-0.5, -0.5, 0);

//
// In- and outputs
//
in vec3 position;

void main()
{
  gl_Position = projMat * viewMat * modelMat * vec4(position, 1.0);
}