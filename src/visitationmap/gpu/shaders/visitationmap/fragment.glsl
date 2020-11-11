#version 430

//!Changing this definition also requires changing the definition in the shader code!
#define NUMBER_OF_REPRESENTATIVE_FIBERS 25

//
//
// Structs
//
//
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
const float stepSize = .5;
const float gradCalcRadius = 3; //0.75;

const float INF_POS =  1. / 0.; //works from OpenGL 4.1 and on
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

uniform bool useFrequencyIsovalue;
uniform uint frequencyIsovalueThreshold;
uniform double maxDistanceScoreIsovalueThreshold;
uniform VisitationMapProperties vmp; //visitationMapProp

uniform bool useInterpolation;

//
//
// SSBOs
//
//
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

//
//
// Functions
//
//
uint GetCellIndex(in uint x_index, in uint y_index, in uint z_index)
{
    return x_index + vmp.width * (y_index + z_index * vmp.height);
}

vec3 GetPosition(in uint x_index, in uint y_index, in uint z_index)
{
    return vec3(
        (vmp.dataset_aabb.xmin + int(x_index)) * vmp.cellSize,
        (vmp.dataset_aabb.ymin + int(y_index)) * vmp.cellSize,
        (vmp.dataset_aabb.zmin + int(z_index)) * vmp.cellSize
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

bool isMinimumDistanceScoreLowerThanThreshold(in uint cellIndex)
{
    Cell cell = cells[cellIndex];

    uint fiberId = cell.representativeFibers[0];

    if(fiberId == 0) //There is no fiber in the representative fibers list
    {
        return false;
    }

    double distanceScore = distanceScores[fiberId];

//    TODO: This was a temproary hack to avoid flickering and glitches longer into the the rendering
//    if(distanceScore <= 0)
//    {
//        return false;
//    }

    if(distanceScore <= maxDistanceScoreIsovalueThreshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool isVoxelInIsosurface(in uint cellIndex)
{
    //Out of bounds checking
    if(cellIndex > vmp.width * vmp.height * vmp.depth)
    {
        return false;
    }

    if(useFrequencyIsovalue)
    {
        uint isovalue = cells[cellIndex].numberOfFibers;
        return isovalue > frequencyIsovalueThreshold;//TODO: should this be geq?
    }
    else
    {
        return isMinimumDistanceScoreLowerThanThreshold(cellIndex);
    }
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

    vec3 voxelStep = 1/2 * vmp.cellSize * eyePosVec;

    return !isVoxelInIsosurface(position + voxelStep);
}

vec3 computeNormal(in vec3 position)
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

vec4 computeShading(in vec3 position, in vec3 eyeVec)
{
    vec3 normal = computeNormal(position);

    //Surface material properties
    float k_a = 0.3;  //ambient
    float k_d = 0.7;  //diffuse
    float k_s = 0.2;  //specular
    float alpha = 5; //shininess

    //Light properties
    vec3 i_a = vec3(1, 1, 0);
    vec3 i_d = vec3(1, 1, 0);
    vec3 i_s = vec3(1);

    vec3 color = vec3(0);

    color += k_a * i_a;                       //ambient contribution
    color += k_d * dot(eyeVec, normal) * i_d; //diffuse contribution

//    vec3 R_m = 2 * dot(eyeVec, normal) * normal - eyeVec; //perfect reflection direction
//    color += k_s * pow(dot(R_m, eyeVec), alpha) * i_s; //specular contribution

    return vec4(color, 1);
}

float trilinearInterpolation(in vec3 position)
{
    uint x_index, y_index, z_index;
    GetIndices(position, x_index, y_index, z_index);

    vec3 P000 = GetPosition(x_index,     y_index,     z_index    );
    vec3 P100 = GetPosition(x_index + 1, y_index,     z_index    );
    vec3 P010 = GetPosition(x_index,     y_index + 1, z_index    );
    vec3 P001 = GetPosition(x_index,     y_index,     z_index + 1);
    vec3 P101 = GetPosition(x_index + 1, y_index,     z_index + 1);
    vec3 P011 = GetPosition(x_index,     y_index + 1, z_index + 1);
    vec3 P110 = GetPosition(x_index + 1, y_index + 1, z_index    );
    vec3 P111 = GetPosition(x_index + 1, y_index + 1, z_index + 1);

    uint c000 = cells[GetCellIndex(x_index,     y_index,     z_index    )].numberOfFibers;
    uint c100 = cells[GetCellIndex(x_index + 1, y_index,     z_index    )].numberOfFibers;
    uint c010 = cells[GetCellIndex(x_index,     y_index + 1, z_index    )].numberOfFibers;
    uint c001 = cells[GetCellIndex(x_index,     y_index,     z_index + 1)].numberOfFibers;
    uint c101 = cells[GetCellIndex(x_index + 1, y_index,     z_index + 1)].numberOfFibers;
    uint c011 = cells[GetCellIndex(x_index,     y_index + 1, z_index + 1)].numberOfFibers;
    uint c110 = cells[GetCellIndex(x_index + 1, y_index + 1, z_index    )].numberOfFibers;
    uint c111 = cells[GetCellIndex(x_index + 1, y_index + 1, z_index + 1)].numberOfFibers;

    float x_d = (position.x - P000.x) / (P111.x - P000.x);
    float y_d = (position.y - P000.y) / (P111.y - P000.y);
    float z_d = (position.z - P000.z) / (P111.z - P000.z);

    float c00 = c000 * (1 - x_d) + c100 * x_d;
    float c01 = c001 * (1 - x_d) + c101 * x_d;
    float c10 = c010 * (1 - x_d) + c110 * x_d;
    float c11 = c011 * (1 - x_d) + c111 * x_d;

    float c0 = c00 * (1 - y_d) + c10 * y_d;
    float c1 = c01 * (1 - y_d) + c11 * y_d;

    float c = c0 * (1 - z_d) + c1 * z_d;

    return c;
}

float calculateIsovalue(in vec3 positionWC)
{
    float isovalue;

    //if(useInterpolation)
    if(true)
    {
        isovalue = trilinearInterpolation(positionWC);
    }
    else
    {
        uint cellIndex = GetCellIndex(positionWC);
        isovalue = cells[cellIndex].numberOfFibers;
    }

    return isovalue;
}

//Tterative  bisection procedure
//Based on https://cgl.ethz.ch/people/archive/siggc/publications/eg05.pdf
const uint numberOfRefinementIterationSteps = 4;
void intersectionRefinement(in vec3 x_near, in vec3 x_far, out vec3 refinedIntersection, out float isovalue)
{
    vec3 x_new;
    float f_near, f_far, f_new;

    f_near = calculateIsovalue(x_near);
    f_far = calculateIsovalue(x_far);

    for(uint i = 0; i < numberOfRefinementIterationSteps; i++)
    {
        x_new = (f_far - f_near) * (frequencyIsovalueThreshold - f_near) / (f_far - f_near) + x_near;
        f_new = calculateIsovalue(x_new);

        if(f_new > frequencyIsovalueThreshold)
        {
            // new point lies in front of the isosurface
            //todo I THINK IT'S THE OTHER WAY AROUND? BECAUSE WE USE THE INVERSE OF THE DEFINITION OF ISOVALUES?

            x_near = x_new;
            f_near = f_new;
        }
        else
        {
            x_far = x_new;
            f_far = f_new;
        }
    }

    refinedIntersection = x_new;
    isovalue = f_new;
}

bool nearIsosurface(in uint x_index, in uint y_index, in uint z_index)
{
    return (
        cells[GetCellIndex(x_index + 1, y_index,     z_index    )].numberOfFibers >= frequencyIsovalueThreshold
    ||  cells[GetCellIndex(x_index,     y_index + 1, z_index    )].numberOfFibers >= frequencyIsovalueThreshold
    ||  cells[GetCellIndex(x_index,     y_index,     z_index + 1)].numberOfFibers >= frequencyIsovalueThreshold
    ||  cells[GetCellIndex(x_index - 1, y_index,     z_index    )].numberOfFibers >= frequencyIsovalueThreshold
    ||  cells[GetCellIndex(x_index,     y_index - 1, z_index    )].numberOfFibers >= frequencyIsovalueThreshold
    ||  cells[GetCellIndex(x_index,     y_index,     z_index - 1)].numberOfFibers >= frequencyIsovalueThreshold
//    ||  cells[GetCellIndex(x_index + 1, y_index,     z_index + 1)].numberOfFibers
//    ||  cells[GetCellIndex(x_index,     y_index + 1, z_index + 1)].numberOfFibers
//    ||  cells[GetCellIndex(x_index + 1, y_index + 1, z_index    )].numberOfFibers
//    ||  cells[GetCellIndex(x_index + 1, y_index + 1, z_index + 1)].numberOfFibers
    );
}

void getCellMinMax(in uint x_index, in uint y_index, in uint z_index, out vec3 cellMin, out vec3 cellMax)
{
    vec3 cellCenter = GetPosition(x_index, y_index, z_index);
    float halfSize = vmp.cellSize / 2.0f;

    cellMin = cellCenter - vec3(halfSize, halfSize, halfSize);
    cellMax = cellCenter + vec3(halfSize, halfSize, halfSize);
}

//
//
// Main loop
//
//
void main()
{
    vec3 eyePosVec = fragmentPositionWC - cameraPosition;
    vec3 stepDir = normalize(eyePosVec);
    vec3 stepVec = stepSize * stepDir;

    vec4 fragmentColor = vec4(0);

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
        if(s > tFar)
        {
            //We exit the ROI, so we stop the raycasting.
            break;
        }

        uint x_index = 0; uint y_index = 0; uint z_index = 0;
        GetIndices(currentPosition, x_index, y_index, z_index);
        uint cellIndex = GetCellIndex(x_index, y_index, z_index);

        //INTERPOLATION MODE:
        //if one of the cells around the sampling point has an isovalue >= threshold, but we were not in the isovalue already,
        //      set 'in_isovalue' to true
        //

        if(!inVolume && nearIsosurface(x_index, y_index, z_index))// isVoxelInIsosurface(currentPosition))
        {
            vec3 cellMin, cellMax;
            getCellMinMax(x_index, y_index, z_index, cellMin, cellMax);
            BoxIntersection intersection = intersectAABB(cameraPosition, stepDir, cellMin, cellMax);

            vec3 x_near = cameraPosition + intersection.Near * stepDir;
            vec3 x_far  = cameraPosition + intersection.Far * stepDir;

            vec3 refinedIntersection; float isovalue;
            intersectionRefinement(x_near, x_far, refinedIntersection, isovalue);

            if(isovalue >= frequencyIsovalueThreshold)
            {
                inVolume = true;
//                fragmentColor += computeShading(refinedIntersection, -stepDir);
                fragmentColor += vec4(1);
                gl_FragDepth = 0.5; //not correct, but works for now;
            }

//            TODO: properly implement depth values
//            vec4 depth_vec = viewMat * projMat * vec4(currentPosition, 1.0);
//            float depth = ((depth_vec.z / depth_vec.w) + 1.0) * 0.5;
//            gl_FragDepth = depth;
        }
//        else if(inVolume)
//        {
//
//        }
        else
        {
            gl_FragDepth = 1;
        }

        currentPosition += stepVec;
        s += stepSize;
    }
    
    outColor = fragmentColor;
}
