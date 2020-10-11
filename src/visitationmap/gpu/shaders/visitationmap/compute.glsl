#version 430 core
layout(local_size_x = 1) in;
//layout(rgba32f, binding = 0) uniform image2D img_output;

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

//layout(std430, binding = 1) buffer FiberSample
//{
//    vec4 vertices[]; //vertex is a vec3 with an empty float for required padding
//};
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
    //TODO: Check if we want to set this as a uniform
//    float xmin = -1/2.0 * vmp.width  * vmp.cellSize;
//    float ymin = -1/2.0 * vmp.height * vmp.cellSize;
//    float zmin = -1/2.0 * vmp.depth  * vmp.cellSize;

    //Casting to uint automatically floors the float
    x_index = uint((point.x - vmp.xmin) / vmp.cellSize);
    y_index = uint((point.y - vmp.ymin) / vmp.cellSize);
    z_index = uint((point.z - vmp.zmin) / vmp.cellSize);
}

void main()
{
    for(int i = 0; i < 100; i++)
    {
        atomicAdd(frequency_map[i], 1);
    }
    //atomicAdd(frequency_map[gl_GlobalInvocationID.x], 1);

//    for(int i = 0; i < visitationMapProp.width * visitationMapProp.height * visitationMapProp.depth; i++)
//    {
//        atomicAdd(frequency_map[i], 1);
////        frequency_map[i] = 3;//frequency_map[i] + 1;;
//    }
}