//VTK::System::Dec // always start with this line
//VTK::Output::Dec // always have this line in your FS
uniform vec3 diffuseColorUniform;

varying vec3 vPos;

//in cell
//{
//  int x;
//  int y;
//  int z;
//} test[1024];

void main ()
{
  //texture3D(cells, )
//  int x = test[0].x;
//  int y = test[0].y;
//  int z = test[0].z;
//  int x = cells[0];
//  int y = cells[1];
//  int z = cells[2];
//  int x = 0;
//  int y = 1;
//  int z = 0;

//  if(abs(vPos.x - x) < 0.3
//  && abs(vPos.y - y) < 0.3
//  && abs(vPos.z - z) < 0.3)
//  {
//    gl_FragData[0] = vec4(0, 1, 0, 1);
//  }
//  else
//  {
    gl_FragData[0] = vec4(1, 0, 0, 1);
//  }
}