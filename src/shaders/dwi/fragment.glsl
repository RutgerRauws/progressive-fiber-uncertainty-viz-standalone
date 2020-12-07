#version 430

//
// Uniforms
//
uniform sampler2D texture;
uniform bool showSlice;

//
// In- and outputs
//
in vec2  TexCoord;
out vec4 FragColor;

void main()
{
  if(!showSlice)
  {
    discard;
  }

  vec4 pixelColor = texture2D(texture, TexCoord);
  FragColor.rgb   = pixelColor.rgb;
  FragColor.a     = 1.0f;
}