#version 430

//!Changing this definition also requires changing the definition in the shader code!
#define NUMBER_OF_REPRESENTATIVE_FIBERS 5

//
//
// Structs
//
//
struct AxisAlignedBoundingBox
{
    int xmin, xmax, ymin, ymax, zmin, zmax;
};

struct VisitationMapProperties
{
    AxisAlignedBoundingBox dataset_aabb; //this AABB is set once and defines the bounds of the DWI/DTI dataset
    float cellSize;
    uint width, height, depth;
};

struct Bucket
{
    uint numberOfFibers;
    uint representativeFibers[NUMBER_OF_REPRESENTATIVE_FIBERS];
};

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
//Choose stepsize of less than or equal to 1.0 voxel units (or we may get aliasing in the ray direction)
//https://www3.cs.stonybrook.edu/~qin/courses/visualization/visualization-surface-rendering-with-polygons.pdf
const float stepSize = .05;

const float INF_POS =  1. / 0.;
const float INF_NEG = -1. / 0.;

//
//
// Uniforms
//
//
uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

uniform vec3 cameraPosition;

uniform float isovalueThreshold;
uniform VisitationMapProperties vmp; //visitationMapProp

//
//
// SSBOs
//
//
layout(std430, binding = 0) buffer visitationMap
{
    uint multiMapIndices[];
};

layout(std430, binding = 1) buffer regionsOfInterest
{
    AxisAlignedBoundingBox ROIs[]; //these AABBBs will continuously change during execution, when new fibers are added
};

layout(std430, binding = 3) buffer CellFiberMultiMap
{
    uint numberOfBucketsUsed;
    Bucket buckets[];
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
    x_index = uint((point.x - vmp.dataset_aabb.xmin * vmp.cellSize) / vmp.cellSize);
    y_index = uint((point.y - vmp.dataset_aabb.ymin * vmp.cellSize) / vmp.cellSize);
    z_index = uint((point.z - vmp.dataset_aabb.zmin * vmp.cellSize) / vmp.cellSize);
}

uint GetCellIndex(in vec3 positionWC)
{
    uint x_index, y_index, z_index;
    GetIndices(positionWC, x_index, y_index, z_index);

    return GetCellIndex(x_index, y_index, z_index);
}

//https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
vec2 intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax)
{
    vec3 tMin = (boxMin - rayOrigin) / rayDir;
    vec3 tMax = (boxMax - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

vec2 intersectAABBs(vec3 rayOrigin, vec3 rayDir)
{
    vec2 t = vec2(INF_POS, INF_NEG);
                                                                                            //todo: HIER VERDER KIJKEN, AABBs zijn nog niet goed geinitieerd voor intersect()??
    for(uint i = 0; i < ROIs.length(); i++)
    {
        AxisAlignedBoundingBox aabb = ROIs[i];

        vec2 t_local = intersectAABB(
            rayOrigin,
            rayDir,
            vec3(aabb.xmin * vmp.cellSize, aabb.ymin * vmp.cellSize, aabb.zmin * vmp.cellSize),
            vec3(aabb.xmax * vmp.cellSize, aabb.ymax * vmp.cellSize, aabb.zmax * vmp.cellSize)
        );

        float tNear = t_local.x;
        float tFar = t_local.y;
        if(tNear > tFar || (tNear < 0 && tFar < 0)) { continue; }

        t.x = min(t.x, t_local.x);
        t.y = max(t.y, t_local.y);
    }

    return t;
}

bool InAABB(in AxisAlignedBoundingBox aabb, in vec3 position)
{
    return (
           position.x >= aabb.xmin * vmp.cellSize
        && position.x <= aabb.xmax * vmp.cellSize
        && position.y >= aabb.ymin * vmp.cellSize
        && position.y <= aabb.ymax * vmp.cellSize
        && position.z >= aabb.zmin * vmp.cellSize
        && position.z <= aabb.zmax * vmp.cellSize
    );
}

bool InAABBs(in vec3 position)
{
    for(uint i = 0; i < ROIs.length(); i++)
    {
        AxisAlignedBoundingBox aabb = ROIs[i];

        bool result = InAABB(aabb, position);

        if(result) { return true; }
    }

    return false;
}

bool isVoxelInIsosurface(in uint cellIndex)
{
    //Out of bounds checking
    if(cellIndex > vmp.width * vmp.height * vmp.depth)
    {
        return false;
    }

    uint multiMapIndex = multiMapIndices[cellIndex];

    if(multiMapIndex == 0) { return false; }

    uint isovalue = buckets[multiMapIndex].numberOfFibers;

    return isovalue > isovalueThreshold; //TODO: should this be geq?
}

bool isVoxelInIsosurface(in vec3 position)
{
    uint cellIndex = GetCellIndex(position);
    return isVoxelInIsosurface(cellIndex);
}


bool isVoxelVisible(in vec3 position)
{
    //todo: verify that this is correct
    vec3 eyePosVec = normalize(cameraPosition - fragmentPositionWC);

    vec3 voxelStep = float(vmp.cellSize) * eyePosVec;

    return !isVoxelInIsosurface(position + voxelStep);
}

vec3 computeNormal(in vec3 position)
{
    vec3 normal = vec3(0);
    float rad = 5;

    for(float x = position.x - rad; x < position.x + rad; x += vmp.cellSize)
    {
        for(float y = position.y - rad; y < position.y + rad; y += vmp.cellSize)
        {
            for(float z = position.z - rad; z < position.z + rad; z += vmp.cellSize)
            {
                vec3 nextVoxel = vec3(x, y, z);

                if(isVoxelVisible(nextVoxel)) {        // isVoxelVisible() just checks if the voxel in question is exposed on the surface (not covered up)
//                    uint cellIndex = GetCellIndex(nextVoxel);
//                    uint value = frequency_map[cellIndex];
//                    normal += value * normalize(nextVoxel - position);
                    normal += normalize(nextVoxel - position);
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

//
//
// Main loop
//
//
void main ()
{
    vec3 eyePosVec = fragmentPositionWC - cameraPosition;
    vec3 stepDir = normalize(eyePosVec);
    vec3 stepVec = stepSize * stepDir;

    vec4 fragmentColor = vec4(0);

    //Find start point
    vec2 t = intersectAABBs(
        cameraPosition,
        stepDir
    );

    float tNear = t.x;
    float tFar = t.y;

    //If tNear > tFar, then there is no intersection
    //If tNear and tFar are negative, the ROI is behind the camera
    if(tNear > tFar || (tNear < 0 && tFar < 0)) { outColor = fragmentColor; return; }

    vec3 currentPosition;

    //Todo: do we need this still?
    if(InAABBs(cameraPosition))
    {
        currentPosition = cameraPosition;
    }
    else
    {
        currentPosition = cameraPosition + tNear * stepDir;
    }

    float s = tNear;

    //Start ray traversal
    while(fragmentColor.w < 1.0f)
    {
        if(s > tFar)
        {
            //We exit the ROI, so we stop the raycasting.
            break;
        }

        if(isVoxelInIsosurface(currentPosition))
        {
//            vec3 normal = computeNormal(currentPosition);
//            fragmentColor = vec4(normal, 1);
            fragmentColor += vec4(1, 0, 0, 1);
        }

        currentPosition += stepVec;
        s += stepSize;
    }

    outColor = fragmentColor;
}