#version 430 core
layout(local_size_x = 1) in;
//layout(rgba32f, binding = 0) uniform image2D img_output;

//#extension GL_ARB_compute_shader : enable
//#extension GL_ARB_compute_variable_group_size : enable

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
    AxisAlignedBoundingBox roi_aabb; //this AABBB will continuously change during execution, when new fibers are added
    uint frequency_map[];
};

layout(std430, binding = 1) buffer FiberSegments
{
//    uint numberOfVertices; //let's use vertices.length for now
    vec4 vertices[]; //vertex is a vec3 with an empty float for required padding
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

//todo: change aabb to different variable name than global aabb
//bool InAABB(in AxisAlignedBoundingBox aabb, in vec3 position)
//{
//    return (
//        position.x >= aabb.xmin
//        && position.x <= aabb.xmax
//        && position.y >= aabb.ymin
//        && position.y <= aabb.ymax
//        && position.z >= aabb.zmin
//        && position.z <= aabb.zmax
//    );
//}

void updateROIAABB(in vec3 position)
{
    atomicMin(roi_aabb.xmin, int(floor(position.x / vmp.cellSize)));
    atomicMax(roi_aabb.xmax, int(ceil( position.x / vmp.cellSize)));
    atomicMin(roi_aabb.ymin, int(floor(position.y / vmp.cellSize)));
    atomicMax(roi_aabb.ymax, int(ceil( position.y / vmp.cellSize)));
    atomicMin(roi_aabb.zmin, int(floor(position.z / vmp.cellSize)));
    atomicMax(roi_aabb.zmax, int(ceil( position.z / vmp.cellSize)));
//    roi_aabb.xmin = min(roi_aabb.xmin, position.x);
//    roi_aabb.xmax = max(roi_aabb.xmax, position.x);
//    roi_aabb.ymin = min(roi_aabb.ymin, position.y);
//    roi_aabb.ymax = max(roi_aabb.ymax, position.y);
//    roi_aabb.zmin = min(roi_aabb.zmin, position.z);
//    roi_aabb.zmax = max(roi_aabb.zmax, position.z);
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

    for(float s = 0; s < length; s += vmp.cellSize)
    {
        vec3 currentPos = p1 + s * directionVec;
        uint cellIndex = GetCellIndex(currentPos);

        atomicAdd(frequency_map[cellIndex], 10);
    }
//    uint cellIndex1 = GetCellIndex(p1);
//    uint cellIndex2 = GetCellIndex(p2);
//
//    atomicAdd(frequency_map[cellIndex1], 1);
//    atomicAdd(frequency_map[cellIndex2], 1);
}

//
//
// Main loop
//
//
void main()
{
    if(vertices.length() < 1) { return; }
//    makeSphere();

//    uint numberOfEdges = vertices.length() - 1;
//    uint numberOfSegmentsToCompute = numberOfEdges / gl_NumWorkGroups.x;

    uint numberOfLineSegments = gl_NumWorkGroups.x;
    uint segmentId = gl_WorkGroupID.x; //the segment id is the vertex number for the 'start vertex'

    vec3 currentPoint = vertices[2*segmentId].xyz;
    vec3 nextPoint = vertices[2*segmentId + 1].xyz; //we will not index out of bounds, as the numberOfEdges is always #points-1

//    updateROIAABB(currentPoint);
    updateROIAABB(nextPoint);

    splatLineSegment(currentPoint, nextPoint);

//    for(int i = 0; i < vertices.length() - 1; i += 1)
//    {
//        vec3 currentPoint = vertices[i].xyz;
//        vec3 nextPoint = vertices[i + 1].xyz;
//
//        updateROIAABB(nextPoint);
//
//        splatLineSegment(currentPoint, nextPoint);
//    }

//    for(int x = 0; x < 100; x++)
//    {
//        for(int y = 0; y < 20; y++)
//        {
//            vec3 position = vec3(vmp.zmin + x, vmp.ymin + y, vmp.zmin + 3);
//            uint cellIndex = GetCellIndex(position);
//
//            atomicAdd(frequency_map[cellIndex], 1);
//        }
//    }

    //atomicAdd(frequency_map[gl_GlobalInvocationID.x], 1);

//    for(int i = 0; i < visitationMapProp.width * visitationMapProp.height * visitationMapProp.depth; i++)
//    {
//        atomicAdd(frequency_map[i], 1);
////        frequency_map[i] = 3;//frequency_map[i] + 1;;
//    }
}