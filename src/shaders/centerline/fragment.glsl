#version 430

//tubeFilter->SetRadius(.25); //default is .5
//tubeFilter->SetNumberOfSides(6);

uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

uniform bool showFibers;
uniform vec3 color;
uniform vec3 cameraPosition;

varying vec4 fragmentPositionWC;
varying vec3 Normal;
out vec4 outColor;

vec4 computeShading(in vec3 position, in vec3 eyeVec)
{
    //Surface material properties
    vec3 k_a = color;  //ambient
    vec3 k_d = vec3(0.4, 0.4, 0.2);  //diffuse
    vec3 k_s = vec3(1.0);  //specular
    float alpha = 5; //shininess

    //Light properties
    float i_a = 0.8f;
    float i_d = 0.8f;
    float i_s = 0.3f;

    vec3 outputColor = vec3(0);

    outputColor += k_a * i_a;                       //ambient contribution
    outputColor += k_d * dot(eyeVec, Normal) * i_d; //diffuse contribution

    vec3 R_m = 2 * dot(-eyeVec, Normal) * Normal + eyeVec; //perfect reflection direction
    outputColor += k_s * pow(dot(R_m, eyeVec), alpha) * i_s; //specular contribution

    return vec4(outputColor, 1);
}


void main()
{
    if(showFibers)
    {
        vec3 eyePosDir = -normalize(fragmentPositionWC.xyz - cameraPosition);
        outColor = computeShading(fragmentPositionWC.xyz, eyePosDir);

        gl_FragDepth = ((fragmentPositionWC.z / fragmentPositionWC.w) + 1.0) * 0.5;
    }
    else
    {
        gl_FragDepth = 1.0f;
        discard;
    }
}