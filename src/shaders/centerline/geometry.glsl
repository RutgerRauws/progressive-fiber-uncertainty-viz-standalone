#version 430

uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

uniform float radius;

const uint nrOfSides = 8;
const float PI = 3.1415926538;
const float tDiff = PI / nrOfSides;

layout (lines) in;
layout (triangle_strip, max_vertices = 60) out;

out vec3 Normal;
out vec4 fragmentPositionWC;

void EmitSide(in vec4 p1, in vec4 p2, in vec4 p3, in vec4 p4, in vec4 center)
{
    gl_Position = projMat * viewMat * modelMat * p1;
    fragmentPositionWC = gl_Position;
    Normal = normalize(p1 - center).xyz;
    EmitVertex();

    gl_Position = projMat * viewMat * modelMat * p2;
    fragmentPositionWC = gl_Position;
    Normal = normalize(p2 - center).xyz;
    EmitVertex();

    gl_Position = projMat * viewMat * modelMat * p3;
    fragmentPositionWC = gl_Position;
    Normal = normalize(p1 - center).xyz;
    EmitVertex();

    EndPrimitive();


    gl_Position = projMat * viewMat * modelMat * p3;
    fragmentPositionWC = gl_Position;
    Normal = normalize(p1 - center).xyz;
    EmitVertex();

    gl_Position = projMat * viewMat * modelMat * p4;
    fragmentPositionWC = gl_Position;
    Normal = normalize(p2 - center).xyz;
    EmitVertex();

    gl_Position = projMat * viewMat * modelMat * p2;
    fragmentPositionWC = gl_Position;
    Normal = normalize(p2 - center).xyz;
    EmitVertex();

    EndPrimitive();
}

vec3 GetAxis1(in vec4 directionVector)
{
    vec3 result = cross(directionVector.xyz, vec3(1, 0, 0));

    if(result.x == 0 && result.y == 0 && result.z == 0)
    {
        result = cross(directionVector.xyz, vec3(0, 1, 0));
    }

    return normalize(result);
}

vec3 GetAxis2(in vec4 directionVector)
{
    vec3 result = cross(directionVector.xyz, vec3(0, 0, 1));

    if(result.x == 0 && result.y == 0 && result.z == 0)
    {
        result = cross(directionVector.xyz, vec3(0, 1, 0));
    }

    return normalize(result);
}

vec3 GetPointOnCircle(in vec3 x, in vec3 y, in float t)
{
    return radius * (y * sin(t) + x * cos(t));
}

void main()
{
    vec4 startPoint = gl_in[0].gl_Position;
    vec4 endPoint   = gl_in[1].gl_Position;

    vec4 lineDirection = vec4(normalize(endPoint - startPoint).xyz, 0); //z
//    vec4 lineDirection = normalize(endPoint - startPoint);

    vec3 x = GetAxis1(lineDirection);
    vec3 y = GetAxis2(lineDirection);

    for(uint i = 0; i < nrOfSides; i++)
    {
        float t1 = (i % nrOfSides) * tDiff;
        vec4 circleOffset1 = vec4(GetPointOnCircle(x, y, t1), 0);

        float t2 = ((i + 1) % nrOfSides) * tDiff;
        vec4 circleOffset2 = vec4(GetPointOnCircle(x, y, t2), 0);

        EmitSide(
            startPoint + circleOffset1 - 0.1 * lineDirection,
            startPoint + circleOffset2 - 0.1 * lineDirection,
            endPoint   + circleOffset1 + 0.1 * lineDirection,
            endPoint   + circleOffset2 + 0.1 * lineDirection,
            startPoint
        );
    }
}
