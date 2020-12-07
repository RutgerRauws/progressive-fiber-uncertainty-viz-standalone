#version 430

//!Changing this definition also requires changing the definition in the shader code!
#define NUMBER_OF_REPRESENTATIVE_FIBERS 25

/*
 *
 * STRUCTS
 *
 */
struct AxisAlignedBoundingBox
{
    int xmin, xmax, ymin, ymax, zmin, zmax;
};

struct BoxIntersection
{
    float Near, Far;
};

struct VisitationMapProperties
{
    AxisAlignedBoundingBox dataset_aabb; //this AABB is set once and defines the bounds of the DWI/DTI dataset
    float cellSize;
    uint width, height, depth;
};

struct Cell
{
    uint numberOfFibers;
    uint representativeFibers[NUMBER_OF_REPRESENTATIVE_FIBERS];
};

/*
 *
 * Inputs and outputs
 *
 */
in vec3 fragmentPositionWC;
out vec4 outColor;

/*
 *
 * Hardcoded
 *
 */
//Choose stepsize of less than or equal to 1.0 voxel units (or we may get aliasing in the ray direction)
//https://www3.cs.stonybrook.edu/~qin/courses/visualization/visualization-surface-rendering-with-polygons.pdf
const float stepSizeNearest = .5;
const float stepSizeInterpolation = 1;
const float gradCalcRadius = 3; //0.75;

const float INF_POS =  1. / 0.; //works from OpenGL 4.1 and on
const float INF_NEG = -1. / 0.;

/*
 *
 * Uniforms
 *
 */
uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

uniform vec3 cameraPosition;

uniform bool useFrequencyIsovalue;
uniform float frequencyIsovalueThreshold;
uniform double maxDistanceScoreIsovalueThreshold;
uniform VisitationMapProperties vmp; //visitationMapProp

uniform bool useInterpolation;
uniform float opacity;

uniform vec3 k_ambient;
uniform vec3 k_diffuse;
uniform vec3 k_specular;

/*
 *
 *
 * SSBOs
 *
 */
layout(std430, binding = 0) coherent buffer visitationMap
{
    Cell cells[];
};

layout(std430, binding = 1) buffer regionsOfInterest
{
    AxisAlignedBoundingBox ROIs[]; //these AABBBs will continuously change during execution, when new fibers are added
};

layout(std430, binding = 3) buffer DistanceScores
{
    double distanceScores[]; //sorted by fiber IDs in ascending order
};

/*
 *
 * INDEXING HELPER FUNCTIONS
 *
 */
uint GetCellIndex(in uint x_index, in uint y_index, in uint z_index)
{
    return x_index + vmp.width * (y_index + z_index * vmp.height);
}

vec3 GetPosition(in uint x_index, in uint y_index, in uint z_index)
{
    float halfSize = vmp.cellSize / 2.0f;
    return vec3(
        (vmp.dataset_aabb.xmin + int(x_index)) * vmp.cellSize + halfSize,
        (vmp.dataset_aabb.ymin + int(y_index)) * vmp.cellSize + halfSize,
        (vmp.dataset_aabb.zmin + int(z_index)) * vmp.cellSize + halfSize
    );
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

/*
 *
 * AABB HELPER FUNCTIONS
 *
 */
//https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
BoxIntersection intersectAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax)
{
    vec3 tMin = (boxMin - rayOrigin) / rayDir;
    vec3 tMax = (boxMax - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return BoxIntersection(tNear, tFar);
}

BoxIntersection intersectAABBs(vec3 rayOrigin, vec3 rayDir)
{
    BoxIntersection t = BoxIntersection(INF_POS, INF_NEG);

    for(uint i = 0; i < ROIs.length(); i++)
    {
        AxisAlignedBoundingBox aabb = ROIs[i];

        if(aabb.xmin == 0 && aabb.xmax == 0 && aabb.ymin == 0 && aabb.ymax == 0 && aabb.zmin == 0 && aabb.zmax == 0)
        {
            continue;
        }

        BoxIntersection tLocal = intersectAABB(
            rayOrigin,
            rayDir,
            vec3(aabb.xmin * vmp.cellSize, aabb.ymin * vmp.cellSize, aabb.zmin * vmp.cellSize),
            vec3(aabb.xmax * vmp.cellSize, aabb.ymax * vmp.cellSize, aabb.zmax * vmp.cellSize)
        );

        float tNear = tLocal.Near;
        float tFar = tLocal.Far;
        if(tNear > tFar || (tNear < 0 && tFar < 0)) { continue; }

        t.Near = min(t.Near, tLocal.Near);
        t.Far = max(t.Far, tLocal.Far);
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

void getCellMinMax(in uint x_index, in uint y_index, in uint z_index, out vec3 cellMin, out vec3 cellMax)
{
    vec3 cellCenter = GetPosition(x_index, y_index, z_index);
    float halfSize = vmp.cellSize / 2.0f;

    cellMin = cellCenter - vec3(halfSize, halfSize, halfSize);
    cellMax = cellCenter + vec3(halfSize, halfSize, halfSize);
}

/*
 *
 * ISOVALUE HELPER FUNCTIONS
 *
 */
float GetVoxelIsovalue(in uint cellIndex)
{
    if(useFrequencyIsovalue)
    {
        return float(cells[cellIndex].numberOfFibers);
    }
    else
    {
        uint fiberId = cells[cellIndex].representativeFibers[0];

        if(fiberId == 0) //No fiber present
        {
//            return INF_POS;
            return float(maxDistanceScoreIsovalueThreshold + 1);
        }
        else
        {
            return float(distanceScores[fiberId]); //TODO: shouldn't we keep this a double?
        }
    }
}

float GetVoxelIsovalue(in uint x_index, in uint y_index, in uint z_index)
{
    uint cellIndex = GetCellIndex(x_index, y_index, z_index);
    return GetVoxelIsovalue(cellIndex);
}

float trilinearInterpolation(in vec3 position)
{
    uint x_index, y_index, z_index;
    GetIndices(position, x_index, y_index, z_index);

    vec3 cellCenter = GetPosition(x_index, y_index, z_index);
    float halfSize = vmp.cellSize / 2.0f;

    vec3 P000 = cellCenter - vec3(halfSize, halfSize, halfSize);
    vec3 P111 = cellCenter + vec3(halfSize, halfSize, halfSize);

    float c000 = GetVoxelIsovalue(x_index,     y_index,     z_index    );
    float c100 = GetVoxelIsovalue(x_index + 1, y_index,     z_index    );
    float c010 = GetVoxelIsovalue(x_index,     y_index + 1, z_index    );
    float c001 = GetVoxelIsovalue(x_index,     y_index,     z_index + 1);
    float c101 = GetVoxelIsovalue(x_index + 1, y_index,     z_index + 1);
    float c011 = GetVoxelIsovalue(x_index,     y_index + 1, z_index + 1);
    float c110 = GetVoxelIsovalue(x_index + 1, y_index + 1, z_index    );
    float c111 = GetVoxelIsovalue(x_index + 1, y_index + 1, z_index + 1);

    float x_d = (position.x - P000.x) / vmp.cellSize;
    float y_d = (position.y - P000.y) / vmp.cellSize;
    float z_d = (position.z - P000.z) / vmp.cellSize;

    float c00 = c000 * (1 - x_d) + c100 * x_d;
    float c01 = c001 * (1 - x_d) + c101 * x_d;
    float c10 = c010 * (1 - x_d) + c110 * x_d;
    float c11 = c011 * (1 - x_d) + c111 * x_d;

    float c0 = c00 * (1 - y_d) + c10 * y_d;
    float c1 = c01 * (1 - y_d) + c11 * y_d;

    float c = c0 * (1 - z_d) + c1 * z_d;

    return c;
}

bool withinIsosurface(in float isovalue)
{
    if(useFrequencyIsovalue)
    {
        return isovalue >= frequencyIsovalueThreshold;//TODO: should this be geq? Should we do float cast here?
    }
    else
    {
        return isovalue <= maxDistanceScoreIsovalueThreshold;
    }
}

bool isVoxelInIsosurface(in uint cellIndex)
{
    //Out of bounds checking
    if(cellIndex > vmp.width * vmp.height * vmp.depth)
    {
        return false;
    }

    float isovalue = GetVoxelIsovalue(cellIndex);

    return withinIsosurface(isovalue);
}

bool isVoxelInIsosurface(in vec3 position)
{
    uint cellIndex = GetCellIndex(position);
    return isVoxelInIsosurface(cellIndex);
}

bool isPointInIsosurface(in vec3 position)
{
    float isovalue = trilinearInterpolation(position);
    return withinIsosurface(isovalue);
}

bool isVoxelVisible(in vec3 position)
{
    //todo: verify that this is correct
    vec3 eyePosVec = normalize(cameraPosition - fragmentPositionWC);

    vec3 voxelStep = 1/2 * vmp.cellSize * eyePosVec;

    return !isVoxelInIsosurface(position + voxelStep);
}

const float gradientDelta = 1.0f;
bool isPointVisible(in vec3 position)
{
    if(!isPointInIsosurface(position)) { return false; }

    vec3 eyePosVec = normalize(cameraPosition - fragmentPositionWC);
    vec3 step = gradientDelta * eyePosVec;

    return !isPointInIsosurface(position + step);
}

//Iterative  bisection procedure
//Based on https://cgl.ethz.ch/people/archive/siggc/publications/eg05.pdf
const uint numberOfRefinementIterationSteps = 4;
void intersectionRefinement(in vec3 x_near, in vec3 x_far, out vec3 refinedIntersection, out float isovalue)
{
    vec3 x_new;
    float f_near, f_far, f_new;

    f_near = trilinearInterpolation(x_near);
    f_far = trilinearInterpolation(x_far);

    for(uint i = 0; i < numberOfRefinementIterationSteps; i++)
    {
        if(useFrequencyIsovalue)
        {
            x_new = (x_far - x_near) * ((frequencyIsovalueThreshold - f_near) / (f_far - f_near)) + x_near;
        }
        else
        {
            x_new = (x_far - x_near) * ((float(maxDistanceScoreIsovalueThreshold) - f_near) / (f_far - f_near)) + x_near;
        }

        f_new = trilinearInterpolation(x_new);

        if((useFrequencyIsovalue && f_new < frequencyIsovalueThreshold)
        || (!useFrequencyIsovalue && f_new > maxDistanceScoreIsovalueThreshold))
        {
            // new point lies in front of the isosurface

            x_near = x_new;
            f_near = f_new;
        }
        else
        {
            x_far = x_new;
            f_far = f_new;
        }
    }

    refinedIntersection = x_far;
    isovalue = f_far;
}

bool nearIsosurface(in uint x_index, in uint y_index, in uint z_index)
{
    return (
        isVoxelInIsosurface(GetCellIndex(x_index,     y_index,     z_index    ))
    ||  isVoxelInIsosurface(GetCellIndex(x_index + 1, y_index,     z_index    ))
    ||  isVoxelInIsosurface(GetCellIndex(x_index,     y_index + 1, z_index    ))
    ||  isVoxelInIsosurface(GetCellIndex(x_index,     y_index,     z_index + 1))
    ||  isVoxelInIsosurface(GetCellIndex(x_index + 1, y_index,     z_index + 1))
    ||  isVoxelInIsosurface(GetCellIndex(x_index,     y_index + 1, z_index + 1))
    ||  isVoxelInIsosurface(GetCellIndex(x_index + 1, y_index + 1, z_index    ))
    ||  isVoxelInIsosurface(GetCellIndex(x_index + 1, y_index + 1, z_index + 1))
    );
}

/*
 *
 * SHADING RELATED FUNCTIONS
 *
 */
vec3 computeNormalVoxel(in vec3 position)
{
    vec3 normal = vec3(0);

    for(float x = position.x - gradCalcRadius; x < position.x + gradCalcRadius; x += vmp.cellSize)
    {
        for(float y = position.y - gradCalcRadius; y < position.y + gradCalcRadius; y += vmp.cellSize)
        {
            for(float z = position.z - gradCalcRadius; z < position.z + gradCalcRadius; z += vmp.cellSize)
            {
                vec3 nextVoxel = vec3(x, y, z);

                if(isVoxelVisible(nextVoxel))        // isVoxelVisible() just checks if the voxel in question is exposed on the surface (not covered up)
                {
                    normal += normalize(nextVoxel - position);
                }
            }
        }
    }

    return normalize(normal);
}

vec3 computeNormalPoint(in vec3 position)
{
    vec3 normal = vec3(0);

//    float offset = 4*gradientDelta;
//    for(float x = position.x - offset; x < position.x + offset; x += gradientDelta)
//    {
//        for(float y = position.y - offset; y < position.y + offset; y += gradientDelta)
//        {
//            for(float z = position.z - offset; z < position.z + offset; z += gradientDelta)
//            {
//                vec3 nextPoint = vec3(x, y, z);
//
//                if(isPointVisible(nextPoint))        // isPointVisible() just checks if the voxel in question is exposed on the surface (not covered up)
//                {
//                    normal += normalize(nextPoint - position);
//                }
//            }
//        }
//    }

    float g_x = 0;
    float g_y = 0;
    float g_z = 0;

    vec3 d_x = vec3(gradientDelta, 0, 0);
    vec3 d_y = vec3(0, gradientDelta, 0);
    vec3 d_z = vec3(0, 0, gradientDelta);

    const uint nr = 5;

    for(uint i = 0; i < nr; i++)
    {
        g_x = trilinearInterpolation(position - i * d_x) - trilinearInterpolation(position + i * d_x);
        g_y = trilinearInterpolation(position - i * d_y) - trilinearInterpolation(position + i * d_y);
        g_z = trilinearInterpolation(position - i * d_z) - trilinearInterpolation(position + i * d_z);
    }

    g_x = g_x / float(nr);
    g_y = g_y / float(nr);
    g_z = g_z / float(nr);

    normal = vec3(g_x, g_y, g_z);

    return normalize(normal);
}

vec3 computeShading(in vec3 position, in vec3 eyeVec)
{
    vec3 normal;
    if(useInterpolation)
    {
        normal = computeNormalPoint(position);
    }
    else
    {
        normal = computeNormalVoxel(position);
    }

    //Surface material properties
    float alpha = 5; //shininess

    //Light properties
    float i_a = 0.8f;
    float i_d = 0.8f;
    float i_s = 0.3f;

    vec3 color = vec3(0);

    color += k_ambient * i_a;                       //ambient contribution
    color += k_diffuse * dot(eyeVec, normal) * i_d; //diffuse contribution

//    vec3 R_m = 2 * dot(eyeVec, normal) * normal - eyeVec; //perfect reflection direction
//    color += k_s * pow(dot(R_m, eyeVec), alpha) * i_s; //specular contribution

    return color;
}


/*
 *
 * MAIN LOOP
 *
 */
void main()
{
    vec3 eyePosVec = fragmentPositionWC - cameraPosition;
    vec3 stepDir = normalize(eyePosVec);

    float stepSize;
    if(useInterpolation) { stepSize = stepSizeInterpolation; }
    else { stepSize = stepSizeNearest; }

    vec3 stepVec = stepSize * stepDir;

    vec4 fragmentColor = vec4(0);
    float fragmentDepth = 1;;

    //Find start point
    BoxIntersection intersection = intersectAABBs(
        cameraPosition,
        stepDir
    );

    float tNear = intersection.Near;
    float tFar = intersection.Far;

    //If tNear > tFar, then there is no intersection
    //If tNear and tFar are negative, the ROI is behind the camera
    if(tNear > tFar || (tNear < 0 && tFar < 0)) { outColor = fragmentColor; gl_FragDepth = 1; return; }

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

    bool inVolume = false;

    //Start ray traversal
    while(fragmentColor.w < 1.0f)
    {
        if(s > tFar) { break; } //We exit the ROI, so we stop the raycasting.

        uint x_index = 0; uint y_index = 0; uint z_index = 0;
        GetIndices(currentPosition, x_index, y_index, z_index);
        uint cellIndex = GetCellIndex(x_index, y_index, z_index);

        if(!inVolume && nearIsosurface(x_index, y_index, z_index))// isVoxelInIsosurface(currentPosition))
        {
            vec3 cellMin, cellMax;
            getCellMinMax(x_index, y_index, z_index, cellMin, cellMax);
            BoxIntersection intersection = intersectAABB(cameraPosition, stepDir, cellMin, cellMax);

            vec3 x_near = cameraPosition + intersection.Near * stepDir;
            vec3 x_far  = cameraPosition + intersection.Far * stepDir;

            vec3 refinedIntersection; float isovalue;

            if(useInterpolation)
            {
                intersectionRefinement(x_near, x_far, refinedIntersection, isovalue);
            }
            else
            {
                refinedIntersection = currentPosition;// x_near;
                isovalue = GetVoxelIsovalue(cellIndex);
            }

            if(withinIsosurface(isovalue))
            {
                inVolume = true;
                vec3 color = computeShading(refinedIntersection, -stepDir);

                fragmentColor *= 1 - opacity;
                fragmentColor += vec4(color, opacity) * opacity;

                vec4 depth_vec = projMat * viewMat * modelMat * vec4(refinedIntersection, 1.0);
                fragmentDepth = ((depth_vec.z / depth_vec.w) + 1.0) * 0.5;
            }
        }

        currentPosition += stepVec;
        s += stepSize;
    }

    if(inVolume)
    {
        outColor = fragmentColor;
        gl_FragDepth = fragmentDepth;
    }
    else
    {
        gl_FragDepth = 1;
    }
}
