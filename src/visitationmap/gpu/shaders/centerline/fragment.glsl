#version 430

//tubeFilter->SetRadius(.25); //default is .5
//tubeFilter->SetNumberOfSides(6);

uniform bool showFibers;

out vec4 outColor;

void main()
{
    if(showFibers)
    {
        outColor = vec4(1.0, 0, 0, 1);
    }
    else
    {
        outColor = vec4(0.0);
    }
}