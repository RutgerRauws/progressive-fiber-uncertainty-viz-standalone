//
// Created by rutger on 10/27/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_FIBER_MULTI_MAP_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_FIBER_MULTI_MAP_H

#include <GL/gl.h>

class CellFiberMultiMap
{
    private:
        //!Changing these definitions also requires changing the definitions in the shader code!
        static const unsigned int NUMBER_OF_REPRESENTATIVE_FIBERS = 5;
        //static const unsigned int NUMBER_OF_BUCKETS = 10000000; //100000000; //100 million buckets with 4 + 5 * 4 bytes = 2.4 GB

        struct Bucket
        {
            GLuint numberOfFibers = 0;                                      // 4 bytes
            GLuint representativeFibers[NUMBER_OF_REPRESENTATIVE_FIBERS];   // NUMBER_OF_REPRESENTATIVE_FIBERS * 4 bytes
        };

        unsigned int numberOfBuckets;

        GLuint numberOfBucketsUsed = 0;
        Bucket* buckets;

        GLuint ssbo_id;

    public:
        CellFiberMultiMap();
        ~CellFiberMultiMap();

        void Initialize();

        GLuint GetSSBOId() const { return ssbo_id; }
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CELL_FIBER_MULTI_MAP_H
