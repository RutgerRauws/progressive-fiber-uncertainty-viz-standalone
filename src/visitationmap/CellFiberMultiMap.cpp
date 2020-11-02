//
// Created by rutger on 10/27/20.
//

#include <GL/glew.h>
#include <cstring>
#include <iostream>
#include "CellFiberMultiMap.h"

CellFiberMultiMap::CellFiberMultiMap()
{
    GLint size;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &size);

    numberOfBuckets = (size / sizeof(Bucket)) - 10; //remove one bucket to leave room for 'numberOfBucketsUsed' variable
    std::cout << "(using " << numberOfBuckets << " buckets) ... " << std::flush;

    buckets = new Bucket[numberOfBuckets];
    std::memset(buckets, 0, sizeof(Bucket) * numberOfBuckets);

    glGenBuffers(1, &ssbo_id);
}

//CellFiberMultiMap::~CellFiberMultiMap()
//{
//    delete[] buckets;
//}

void CellFiberMultiMap::Initialize()
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint) + sizeof(Bucket) * numberOfBuckets, 0, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &numberOfBucketsUsed);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), sizeof(Bucket) * numberOfBuckets, buckets); //TODO: got a SIGBUS (bus error) here
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    delete[] buckets;
}