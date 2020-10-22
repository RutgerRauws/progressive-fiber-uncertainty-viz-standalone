#version 430

//
//
// Structs
//
//
struct AxisAlignedBoundingBox
{
    float xmin, xmax, ymin, ymax, zmin, zmax;
};

struct VisitationMapProperties
{
    AxisAlignedBoundingBox dataset_aabb; //this AABB is set once and defines the bounds of the DWI/DTI dataset
    float cellSize;
    uint width, height, depth;
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
const float stepSize = .3;
const uint isovalueThreshold = 1;

//
//
// Uniforms
//
//
uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

uniform vec3 cameraPosition;

uniform VisitationMapProperties vmp; //visitationMapProp

//
//
// SSBOs
//
//
layout(std430, binding = 0) buffer visitationMap
{
    AxisAlignedBoundingBox roi_aabb; //this AABBB will continuously change during execution, when new fibers are added
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
    x_index = uint((point.x - vmp.dataset_aabb.xmin) / vmp.cellSize);
    y_index = uint((point.y - vmp.dataset_aabb.ymin) / vmp.cellSize);
    z_index = uint((point.z - vmp.dataset_aabb.zmin) / vmp.cellSize);
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

bool InAABB(in AxisAlignedBoundingBox aabb, in vec3 position)
{
    return (
           position.x >= aabb.xmin
        && position.x <= aabb.xmax
        && position.y >= aabb.ymin
        && position.y <= aabb.ymax
        && position.z >= aabb.zmin
        && position.z <= aabb.zmax
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
    vec3 eyePosVec = normalize(cameraPosition - fragmentPositionWC);

    vec3 voxelStep = float(vmp.cellSize) * eyePosVec;

    return !isVoxelInIsosurface(position + voxelStep);

//
//    vec3 voxelStep_x = vec3(vmp.cellSize, 0, 0);
//    vec3 voxelStep_y = vec3(0, vmp.cellSize, 0);
//    vec3 voxelStep_z = vec3(0, 0, vmp.cellSize);
//
//    //TODO: Fix this function, as right now the empty voxels behind the voxel in question will also allow the voxel
//    //      in question to be considered 'visible'.
//
//    return (
//        !isVoxelInIsosurface(position - voxelStep_x)
//    ||  !isVoxelInIsosurface(position - voxelStep_y)
//    ||  !isVoxelInIsosurface(position - voxelStep_z)
//    ||  !isVoxelInIsosurface(position + voxelStep_x)
//    ||  !isVoxelInIsosurface(position + voxelStep_y)
//    ||  !isVoxelInIsosurface(position + voxelStep_z)
//    );
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
    vec2 t = intersectAABB(
        cameraPosition,
        stepDir,
        vec3(roi_aabb.xmin, roi_aabb.ymin, roi_aabb.zmin),
        vec3(roi_aabb.xmax, roi_aabb.ymax, roi_aabb.zmax)
    );

    float tNear = t.x;
    float tFar = t.y;

    //If tNear > tFar, then there is no intersection
    //If tNear and tFar are negative, the ROI is behind the camera
    if(tNear > tFar || (tNear < 0 && tFar < 0)) { return; }

    vec3 currentPosition;

    if(InAABB(roi_aabb, cameraPosition))
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
//        if(!InAABB(roi_aabb, currentPosition))
//        {
//            fragmentColor += vec4(1, 0, 0, 1);
//            break;
//        }

        if(s > tFar)
        {
            //We exit the ROI, so we stop the raycasting.
            break;
        }

        if(isVoxelInIsosurface(currentPosition))
        {
            vec3 normal = computeNormal(currentPosition);
            fragmentColor = vec4(normal, 1);
//            fragmentColor += vec4(1);
        }

        currentPosition += stepVec;
        s += stepSize;
    }

    outColor = fragmentColor;
}