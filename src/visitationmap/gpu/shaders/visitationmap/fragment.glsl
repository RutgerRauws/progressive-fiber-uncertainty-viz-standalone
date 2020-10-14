#version 430

//
//
// Inputs and outputs
//
//
in vec3 fragmentPositionWC;
out vec4 outColor;

//
//
// Hardcoded
//
//
const float stepSize = 0.1;
const uint isovalueThreshold = 3;

//
//
// Uniforms
//
//
uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

uniform vec3 cameraPosition;

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
    //Casting to uint automatically floors the float
    x_index = uint((point.x - vmp.xmin) / vmp.cellSize);
    y_index = uint((point.y - vmp.ymin) / vmp.cellSize);
    z_index = uint((point.z - vmp.zmin) / vmp.cellSize);
}

uint GetCellIndex(in vec3 positionWC)
{
    uint x_index, y_index, z_index;
    GetIndices(positionWC, x_index, y_index, z_index);

    return GetCellIndex(x_index, y_index, z_index);
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


bool isVoxelInIsosurface(in uint cellIndex)
{
    //Out of bounds checking
    if(cellIndex > vmp.width * vmp.height * vmp.depth)
    {
        return false;
    }

    uint isovalue = frequency_map[cellIndex];
    return isovalue > isovalueThreshold;
}

//bool isVoxelInIsosurface(in uint x_index, in uint y_index, in uint z_index)
//{
//    uint cellIndex = GetCellIndex(x_index, y_index, z_index);
//    return isVoxelInIsosurface(cellIndex);
//}

bool isVoxelInIsosurface(in vec3 position)
{
    uint cellIndex = GetCellIndex(position);
    return isVoxelInIsosurface(cellIndex);
}


bool isVoxelVisible(in vec3 position)
{
    vec3 voxelStep_x = vec3(vmp.cellSize, 0, 0);
    vec3 voxelStep_y = vec3(0, vmp.cellSize, 0);
    vec3 voxelStep_z = vec3(0, 0, vmp.cellSize);

    return (
        !isVoxelInIsosurface(position - voxelStep_x)
    ||  !isVoxelInIsosurface(position - voxelStep_y)
    ||  !isVoxelInIsosurface(position - voxelStep_z)
    ||  !isVoxelInIsosurface(position + voxelStep_x)
    ||  !isVoxelInIsosurface(position + voxelStep_y)
    ||  !isVoxelInIsosurface(position + voxelStep_z)
    );
}

bool isVoxelVisible(in uint cellIndex)
{
    return false;
}

vec3 computeNormal(in vec3 position)
{
    vec3 normal = vec3(0);
    float rad = 5;

    for(float x = position.x - rad; x < position.x + rad; x++)
    {
        for(float y = position.y - rad; y < position.y + rad; y++)
        {
            for(float z = position.z - rad; z < position.z + rad; z++)
            {
                vec3 nextVoxel = vec3(x, y, z);

                if(isVoxelVisible(nextVoxel)) {        // voxelVisible() just checks if the voxel in question is exposed on the surface (not covered up)
                    normal += nextVoxel - position;
                }
            }
        }
    }

    return -normalize(normal);
}

//vec3 computeLighting(in vec3 position)
//{
//
//}

void main ()
{


    vec3 eyePosVec = fragmentPositionWC - cameraPosition;
    vec3 stepDir = normalize(eyePosVec);
    vec3 stepVec = stepSize * stepDir;

    vec4 fragmentColor = vec4(0);
    vec3 currentPosition = fragmentPositionWC + stepVec;

    while(fragmentColor.w < 1.0f)
    {
//        if(length(currentPosition - fragmentPositionWC) > 40)
        if(!InVolume(currentPosition))
        {
//            fragmentColor += vec4(.5, 0, 0, .5);
            break;
        }

        if(isVoxelInIsosurface(currentPosition))
        {
            fragmentColor = vec4(computeNormal(currentPosition), 1);
//            fragmentColor += vec4(0, 1, 0, 1);
        }

        currentPosition += stepVec;
    }

//    if(test)
//    {
//        outColor = vec4(1, 0, 0, 1);
//    }
//    else
//    {
    outColor = fragmentColor;
//    }

//    outColor = vec4(normalize(eyePositionWC), 1);
//    outColor = vec4(normalize(cameraPosition), 1);
}