#version 430 core
layout(local_size_x = 1) in;
//layout(rgba32f, binding = 0) uniform image2D img_output;


struct VisitationMapProperties
{
    int width, height, depth;
    float cellSize;
};
uniform VisitationMapProperties visitationMapProp;

//struct Cell
//{
//    int fiberList[];
//};
//
//layout(std430, binding = 3) buffer visitationMapList
//{
//    Cell cells[];
//};

layout(std430, binding = 0) buffer frequencyMap
{
    uint frequency_map[];
};

void main()
{
    for(int i = 0; i < 10; i++)
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