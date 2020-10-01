//VTK::System::Dec // always start with this line
in vec4 vertexMC;
// use the default normal decl as the mapper
// will then provide the normalMatrix uniform
// which we use later on
//VTK::Normal::Dec
void main () {
// do something weird with the vertex positions
// this will mess up your head if you keep
// rotating and looking at it, very trippy
//            vec4 tmpPos = MCDCMatrix * vertexMC;
//            gl_Position = tmpPos*vec4(0.2+0.8*abs(tmpPos.x),0.2+0.8*abs(tmpPos.y),1.0,1.0);
  gl_Position = vertexMC;
}