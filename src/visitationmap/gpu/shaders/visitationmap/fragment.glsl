#version 430

//
//
// Inputs and outputs
//
//
in vec3 fragmentPosition;
in vec3 eyePosition;
out vec4 outColor;

//
//
// Hardcoded
//
//
const float stepSize = 1;
const uint isovalueThreshold = 3;

//
//
// Uniforms
//
//
struct VisitationMapProperties
{
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double cellSize;
    uint width, height, depth;
};
uniform VisitationMapProperties vmp; //visitationMapProp

//
//
// SSBOs
//
//
layout(std430, binding = 0) buffer frequencyMap
{
    uint frequency_map[];
};

//
//
// Functions
//
//
uint GetCellIndex(in uint x_index, in uint y_index, in uint z_index)
{
    return x_index + vmp.width * (y_index + z_index * vmp.height);
}

void GetIndices(in vec3 point, out uint x_index, out uint y_index, out uint z_index)
{
    //TODO: Check if we want to set this as a uniform

    //Casting to uint automatically floors the float
    x_index = uint((point.x - vmp.xmin) / vmp.cellSize);
    y_index = uint((point.y - vmp.ymin) / vmp.cellSize);
    z_index = uint((point.z - vmp.zmin) / vmp.cellSize);
}

bool InVolume(vec3 position)
{
    return (
           position.x >= vmp.xmin
        && position.x <= vmp.xmax
        && position.y >= vmp.ymin
        && position.y <= vmp.ymax
        && position.z >= vmp.zmin
        && position.z <= vmp.zmax
    );
}

void main ()
{
    vec3 eyePosVec = fragmentPosition - eyePosition;
    vec3 stepDir = -normalize(eyePosVec);
    vec3 stepVec = stepSize * stepDir;

    vec4 fragmentColor = vec4(0);
    vec3 currentPosition = fragmentPosition + stepVec;

    bool test = false;

    while(fragmentColor.w < 1.0f)
    {
        //if(InVolume(currentPosition))
        if(length(currentPosition - fragmentPosition) > 20)
        {
            test = true;
            break;
        }

        uint x_index, y_index, z_index;
        GetIndices(currentPosition, x_index, y_index, z_index);
        uint cellIndex = GetCellIndex(x_index, y_index, z_index);

        uint isovalue = frequency_map[cellIndex];

        if(isovalue > isovalueThreshold)
        {
            fragmentColor += vec4(0.1);
        }

        currentPosition += stepVec;
    }

//    if(test)
//    {
//        outColor = vec4(1, 0, 0, 1);
//    }
//    else
//    {
//        outColor = fragmentColor;
//    }

    outColor = fragmentColor;
}