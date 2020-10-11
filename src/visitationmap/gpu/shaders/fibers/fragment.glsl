#version 430
//#ifndef GL_ES
//#define highp
//#define mediump
//#define lowp
//#endif // GL_ES
//#define attribute in
//#define varying out
////***VT***K::System::Dec // always start with this line
//
////VTK::Output::Dec // always have this line in your FS, VTK uses it to map shader outputs to the framebufer.
//uniform vec3 diffuseColorUniform;
//
//varying vec3 vPos;
//
////layout(std430, binding = 0) buffer frequencyMap
////{
////    uint frequency_map[];
////};
out vec4 outColor;

void main ()
{
//    gl_FragColor = vec4(1, 0, 0, 1);
    outColor = vec4(1.0);
}