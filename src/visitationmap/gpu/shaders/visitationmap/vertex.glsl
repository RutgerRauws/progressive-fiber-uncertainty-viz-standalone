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

//#ifndef GL_ES
//#define highp
//#define mediump
//#define lowp
//#endif // GL_ES
//#define attribute in
//#define varying out
////***VT***K::System::Dec // always start with this line
//
//in vec4 vertexMC;
//// use the default normal decl as the mapper
//// will then provide the normalMatrix uniform
//// which we use later on
////VTK::Normal::Dec
//
//uniform mat4 MCDCMatrix; //Model coordinates to Device coordinates
////out int cells[1024];
//
//varying vec3 vPos;
//
//void main () {
//  vPos = vertexMC.xyz;
//  gl_Position = MCDCMatrix * vertexMC;
//
////  cells[0] = 0;
////  cells[1] = 1;
////  cells[2] = 0;
////
////  glGenBuffers(); // only creates the object's name and a reference to the object
////
////  glBindBuffer(); //set up its internal state by binding to the context
////  glBufferData(); //create mutable storage for a buffer object
////
////  glDeleteBuffers();
//}