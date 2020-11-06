#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

//!Changing this definition also requires changing the definition in the shader code!
#define NUMBER_OF_REPRESENTATIVE_FIBERS 50

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

struct Cell
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
layout(std430, binding = 0) coherent buffer visitationMap
{
    Cell cells[];
};

layout(std430, binding = 1) coherent buffer regionsOfInterest
{
    AxisAlignedBoundingBox ROIs[]; //these AABBBs will continuously change during execution, when new fibers are added
};

layout(std430, binding = 2) buffer FiberSegments
{
    FiberSegment segments[];
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

void GetCellIndices(in uint index, in uint x_index, in uint y_index, in uint z_index)
{
    x_index = index % vmp.width;
    y_index = (index / vmp.width) % vmp.height;
    z_index = (index / vmp.width) / vmp.height;
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
// Cell functions
//
bool isFiberInCell(in Cell cell, in uint fiberId)
{
    for(uint i = 0; i < NUMBER_OF_REPRESENTATIVE_FIBERS; i++)
    {
        if(cell.representativeFibers[i] == fiberId)
        {
            return true;
        }
    }

    return false;
}

void addFiberToCell(in Cell cell, in uint fiberId)
{
    for(uint i = 0; i < cell.representativeFibers.length(); i++)
    {
        if(cell.representativeFibers[i] == 0)
        {
            cell.representativeFibers[i] = fiberId;
            return;
        }
    }
}

void insertIntoMultiMap(in uint cellIndex, in FiberSegment segment)
{
    bool alreadyPresent = false;
    bool inserted = false;

    //insert in open bucket in cell
    for(uint i = 0; i < NUMBER_OF_REPRESENTATIVE_FIBERS; i++)
    {
        if(cells[cellIndex].representativeFibers[i] == segment.fiberId)
        {
            alreadyPresent = true;
            break;
        }

        if(cells[cellIndex].representativeFibers[i] == 0)
        {
            cells[cellIndex].representativeFibers[i] = segment.fiberId;
            inserted = true;
            break;
        }
    }

    if(!alreadyPresent)
    {
        cells[cellIndex].numberOfFibers += 1;
    }

    //todo: evict a more reasonable fiber from the list
    if(!inserted)
    {
        cells[cellIndex].representativeFibers[0] = segment.fiberId;
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

void splatLineSegment(in FiberSegment segment, in uint cellIndex, in vec3 currentPos)
{
    vec3 p1 = segment.p1.xyz;
    vec3 p2 = segment.p2.xyz;

    //todo: think about the consequences for corner points in cylinders not being considered
    if(implicit_cylinder_f(p1, p2, splatRadius, currentPos) <= 0) //point currentPos is on or in cylinder
    {
        insertIntoMultiMap(cellIndex, segment);
        updateROIAABB(segment.seedPointId, currentPos);
    }
}

//
//
// Main loop
//
//
void main()
{
    uint x_index = gl_GlobalInvocationID.x;
    uint y_index = gl_GlobalInvocationID.y;
    uint z_index = gl_GlobalInvocationID.z;

    if(segments.length() < 1
    || x_index >= vmp.width
    || y_index >= vmp.height
    || z_index >= vmp.depth
    )
    { return; }

    uint cellIndex = GetCellIndex(x_index, y_index, z_index);

    vec3 currentPos = GetPosition(x_index, y_index, z_index);

    for (uint i = 0; i < segments.length(); i++)
    {
        splatLineSegment(segments[i], cellIndex, currentPos);
        memoryBarrier(); barrier();
    }
}