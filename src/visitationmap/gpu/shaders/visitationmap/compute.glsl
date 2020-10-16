#version 430 core
layout(local_size_x = 1) in;
//layout(rgba32f, binding = 0) uniform image2D img_output;

//#extension GL_ARB_compute_shader : enable
//#extension GL_ARB_compute_variable_group_size : enable

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

layout(std430, binding = 1) buffer FiberSample
{
    uint numberOfVertices;
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
    x_index = uint((point.x - vmp.xmin) / vmp.cellSize);
    y_index = uint((point.y - vmp.ymin) / vmp.cellSize);
    z_index = uint((point.z - vmp.zmin) / vmp.cellSize);
}

uint GetCellIndex(in vec3 positionWC)
{
    uint x_index, y_index, z_index;
    GetIndices(positionWC, x_index, y_index, z_index);

    return GetCellIndex(x_index, y_index, z_index);
}


void makeSphere()
{
    vec3 centerPointWC = vec3(
        (vmp.xmin + vmp.xmax) / 2.0,
        (vmp.ymin + vmp.ymax) / 2.0,
        (vmp.zmin + vmp.zmax) / 2.0
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

//
//
// Main loop
//
//
void main()
{
    makeSphere();
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