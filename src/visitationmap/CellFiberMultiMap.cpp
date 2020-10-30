//
// Created by rutger on 10/27/20.
//

#include <GL/glew.h>
#include <cstring>
#include "CellFiberMultiMap.h"

CellFiberMultiMap::CellFiberMultiMap()
{
    buckets = new Bucket[NUMBER_OF_BUCKETS];
    // std::fill_n(buckets, NUMBER_OF_BUCKETS, 0);
    std::memset(buckets, 0, sizeof(Bucket) * NUMBER_OF_BUCKETS);

    glGenBuffers(1, &ssbo_id);
}

CellFiberMultiMap::~CellFiberMultiMap()
{
    delete[] buckets;
}

void CellFiberMultiMap::Initialize()
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Bucket) * NUMBER_OF_BUCKETS, buckets, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind
}
