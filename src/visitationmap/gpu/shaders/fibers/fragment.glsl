#version 430

uniform bool showFibers;

out vec4 outColor;

void main()
{
    if(showFibers)
    {
        outColor = vec4(1.0);
    }
    else
    {
        discard;
    }
}