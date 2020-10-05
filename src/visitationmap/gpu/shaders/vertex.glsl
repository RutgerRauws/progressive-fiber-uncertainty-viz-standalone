//VTK::System::Dec // always start with this line
in vec4 vertexMC;
// use the default normal decl as the mapper
// will then provide the normalMatrix uniform
// which we use later on
//VTK::Normal::Dec

uniform mat4 MCDCMatrix; //Model coordinates to Device coordinates
//out int cells[1024];

buffer cell
{
  int x;
  int y;
  int z;
} test[1024];

varying vec3 vPos;

void main () {
  vPos = vertexMC.xyz;
  gl_Position = MCDCMatrix * vertexMC;

//  cells[0] = 0;
//  cells[1] = 1;
//  cells[2] = 0;
//
//  glGenBuffers(); // only creates the object's name and a reference to the object
//
//  glBindBuffer(); //set up its internal state by binding to the context
//  glBufferData(); //create mutable storage for a buffer object
//
//  glDeleteBuffers();
}