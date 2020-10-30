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
    uint frequency_map[];
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
    Bucket buckets[];
};

//struct Cell
//{
//    int fiberList[];
//};
//
//layout(std430, binding = 3) buffer visitationMapList
//{
//    Cell cells[];
//};

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

void makeSphere()
{
    vec3 centerPointWC = vec3(
        ((vmp.dataset_aabb.xmin + vmp.dataset_aabb.xmax) * vmp.cellSize) / 2.0,
        ((vmp.dataset_aabb.ymin + vmp.dataset_aabb.ymax) * vmp.cellSize) / 2.0,
        ((vmp.dataset_aabb.zmin + vmp.dataset_aabb.zmax) * vmp.cellSize) / 2.0
    );

    uint indices[3];
    GetIndices(centerPointWC, indices[0], indices[1], indices[2]);

    float sideSize = min(vmp.width, min(vmp.height, vmp.depth)) / 2.0;
    int sideIndexOffset = int(sideSize / vmp.cellSize) ;
    for(int x = -sideIndexOffset; x < sideIndexOffset; x++)
    {
        for(int y = -sideIndexOffset; y < sideIndexOffset; y++)
        {
            for(int z = -sideIndexOffset; z < sideIndexOffset; z++)
            {
                vec3 newPoint = centerPointWC + vec3(x, y, z);

                if(distance(vec3(0, 0, 0), newPoint) > sideIndexOffset)
                {
                    continue;
                }

                uint cellIndex = GetCellIndex(indices[0] + x, indices[1] + y, indices[2] + z);

                if(cellIndex > vmp.width * vmp.height * vmp.depth)
                {
                    continue;
                }

                atomicAdd(frequency_map[cellIndex], 1);
            }
        }
    }
}

//Todo: do proper convolution-based splatting
void splatLineSegment(in vec3 p1, in vec3 p2)
{
    vec3 directionVec = normalize(p2 - p1);
    float length = distance(p1, p2);


    for(float s = 0; s < length; s += vmp.cellSize / 2.0)
    {
        vec3 currentPos = p1 + s * directionVec;
        uint cellIndex = GetCellIndex(currentPos);

        atomicAdd(frequency_map[cellIndex], 1);
    }

//    uint cellIndex = GetCellIndex(p1);
//    atomicAdd(frequency_map[cellIndex], 10);
//
//    cellIndex = GetCellIndex(p2);
//    atomicAdd(frequency_map[cellIndex], 10);
}

//
//
// Main loop
//
//
void main()
{
    if(segments.length() < 1) { return; }
//    makeSphere();

    uint numberOfLineSegments = gl_NumWorkGroups.x;
    uint segmentId = gl_WorkGroupID.x; //the segment id is the vertex number for the 'start vertex'

    FiberSegment segment = segments[segmentId];

    vec3 currentPoint = segment.p1.xyz;
    vec3 nextPoint = segment.p2.xyz;

    updateROIAABB(segment.seedPointId, currentPoint);
    updateROIAABB(segment.seedPointId, nextPoint);

    splatLineSegment(currentPoint, nextPoint);
}