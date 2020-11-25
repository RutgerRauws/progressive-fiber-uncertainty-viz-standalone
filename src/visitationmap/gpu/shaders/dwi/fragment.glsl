#version 430

//
// Uniforms
//
uniform sampler2D texture;
uniform float opacity;

//
// In- and outputs
//
in vec2  TexCoord;
out vec4 FragColor;

void main()
{
  vec4 pixelColor = texture2D(texture, TexCoord);
  FragColor.rgb   = pixelColor.rgb;
  FragColor.a     = opacity;
}