#version 430

//
// Uniforms
//
uniform sampler2D texture;

//
// In- and outputs
//
in vec2  TexCoord;
out vec4 FragColor;

void main()
{
  vec4 pixelColor = texture2D(texture, TexCoord);

  if(pixelColor.r < 0.1)
  {
    discard;
  }
  else
  {
    FragColor.rgb = pixelColor.rgb;
    FragColor.a = 1.0f;
  }
}