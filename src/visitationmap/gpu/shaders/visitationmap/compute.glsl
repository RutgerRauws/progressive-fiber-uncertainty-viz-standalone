#version 430 core
layout(local_size_x = 1) in;

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

struct FiberSegment
{
    vec4 p1;                 // 16 bytes
    vec4 p2;                 // 16 bytes
    uint seedPointId;        // 4 bytes
    uint fiberId;            // 4 bytes
    uint padding2, padding3; // 8 bytes
};

struct Bucket
{
    uint numberOfFibers;
    uint representativeFibers[NUMBER_OF_REPRESENTATIVE_FIBERS];
};

//
//
// Hardcoded
//
//
const float splatRadius = 2;

//
//
// Uniforms
//
//
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

layout(std430, binding = 2) buffer FiberSegments
{
    FiberSegment segments[];
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

//
// Visitation Map Functions
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

//todo: not used
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

void updateROIAABB(in uint seedPointId, in vec3 position)
{
    atomicMin(ROIs[seedPointId].xmin, int(floor(position.x / vmp.cellSize)));
    atomicMax(ROIs[seedPointId].xmax, int(ceil( position.x / vmp.cellSize)));
    atomicMin(ROIs[seedPointId].ymin, int(floor(position.y / vmp.cellSize)));
    atomicMax(ROIs[seedPointId].ymax, int(ceil( position.y / vmp.cellSize)));
    atomicMin(ROIs[seedPointId].zmin, int(floor(position.z / vmp.cellSize)));
    atomicMax(ROIs[seedPointId].zmax, int(ceil( position.z / vmp.cellSize)));
}

//
// Bucket functions
//
bool isFiberInBucket(in Bucket bucket, in uint fiberId)
{
    for(uint i = 0; i < bucket.representativeFibers.length(); i++)
    {
        if(bucket.representativeFibers[i] == fiberId)
        {
            return true;
        }
    }

    return false;
}

void addFiberToBucket(in Bucket bucket, in uint fiberId)
{
    for(uint i = 0; i < bucket.representativeFibers.length(); i++)
    {
        if(bucket.representativeFibers[i] == 0)
        {
            bucket.representativeFibers[i] = fiberId;
            return;
        }
    }
}

void insertIntoMultiMap(in uint cellIndex, in FiberSegment segment)
{
    uint bucketIndex = multiMapIndices[cellIndex];

    if(bucketIndex == 0) //there is no entry yet
    {
        bucketIndex = atomicAdd(numberOfBucketsUsed, 1) + 1;
        atomicCompSwap(multiMapIndices[cellIndex], 0, bucketIndex);
    }

    if(!isFiberInBucket(buckets[bucketIndex], segment.fiberId))
    {
        atomicAdd(buckets[bucketIndex].numberOfFibers, 1);
        addFiberToBucket(buckets[bucketIndex], segment.fiberId);
    }
}

//
// General functions
//
//implicit definition of a cylinder (http://www.unchainedgeometry.com/jbloom/pdf/imp-techniques.pdf)
//http://cs-people.bu.edu/sbargal/Fall%202016/lecture_notes/Nov_3_3d_geometry_representation
//f(p) = |p - Closest(ab, p) - r
//with r = radius, p = point, ab = segment from point a to b.
//Closest(ab, p) is the distance to the closest point on the line segment:
vec3 Closest(in vec3 a, in vec3 b, in vec3 p)
{
    vec3 d = b - a;
    vec3 u = p - a;
    float alpha = dot(d, u) / dot(d, d);

    if(alpha <= 0)
    {
        return a;
    }
    else if(alpha >= 1)
    {
        return b;
    }
    else //alpha > 0 && alpha < 1
    {
        return a + alpha * d;
    }
}

//returns > 0 if outside, < 0 if inside, = 0 when p is on surface
float implicit_cylinder_f(in vec3 a, in vec3 b, in float radius, in vec3 p)
{
    return length(p - Closest(a, b, p)) - radius;
}

void splatLineSegment(in FiberSegment segment)
{
    vec3 p1 = segment.p1.xyz; vec3 p2 = segment.p2.xyz;

//    vec3 directionVec = normalize(p2 - p1);
//    float length = distance(p1, p2);

    float xmin =  min(p1.x, p2.x) - splatRadius;
    float xmax = max(p1.x, p2.x) + splatRadius;
    float ymin = min(p1.y, p2.y) - splatRadius;
    float ymax = max(p1.y, p2.y) + splatRadius;
    float zmin = min(p1.z, p2.z) - splatRadius;
    float zmax = max(p1.z, p2.z) + splatRadius;

    for(float x = xmin; x <= xmax; x += vmp.cellSize)
    {
        for(float y = ymin; y <= ymax; y += vmp.cellSize)
        {
            for(float z = zmin; z <= zmax; z += vmp.cellSize)
            {
                vec3 currentPos = vec3(x, y, z);
                uint cellIndex = GetCellIndex(currentPos);

                if(implicit_cylinder_f(p1, p2, splatRadius, currentPos) <= 0)
                {
                    insertIntoMultiMap(cellIndex, segment);
                }
            }
        }
    }

    updateROIAABB(segment.seedPointId, vec3(xmin, ymin, zmin));
    updateROIAABB(segment.seedPointId, vec3(xmax, ymax, zmax));

//    for(float s = 0; s < length; s += vmp.cellSize / 2.0)
//    {
//        vec3 currentPos = p1 + s * directionVec;
//        uint cellIndex = GetCellIndex(currentPos);
//
//        if(implicit_cylinder_f(p1, p2, splatRadius, currentPos) <= 0)
//        {
//            insertIntoMultiMap(cellIndex, segment);
//        }
//    }

//    uint cellIndex = GetCellIndex(p1);
//    atomicAdd(multiMapIndices[cellIndex], 10);
//
//    cellIndex = GetCellIndex(p2);
//    atomicAdd(multiMapIndices[cellIndex], 10);
}

//
//
// Main loop
//
//
void main()
{
    if(segments.length() < 1) { return; }

    uint numberOfLineSegments = gl_NumWorkGroups.x;
    uint segmentId = gl_WorkGroupID.x; //the segment id is the vertex number for the 'start vertex'

    FiberSegment segment = segments[segmentId];

    vec3 currentPoint = segment.p1.xyz;
    vec3 nextPoint = segment.p2.xyz;

    splatLineSegment(segment);
}