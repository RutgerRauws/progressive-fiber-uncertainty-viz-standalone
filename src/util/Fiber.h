//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H

#include <vector>
#include <GL/gl.h>
#include "vtkSmartPointer.h"
#include "glm/ext.hpp"

class Fiber
{
    public:
//        class __attribute__((__packed__)) LineSegment
        class __attribute__((__packed__)) LineSegment
        {
            private:
                glm::vec4 p1; //4*4 = 16 bytes
                glm::vec4 p2; //4*4 = 16 bytes
                GLuint seedPointId; // 4 bytes
                GLuint padding1, padding2, padding3; // 12 bytes
//                float p1_x, p1_y, p1_z;
//                float p2_x, p2_y, p2_z;
//                GLuint seedPointId;
//                glm::vec4 p1; //4 * 4 = 16 bytes
//                glm::vec4 p2; //4 * 4 = 16 bytes
//                GLuint seedPointId; //4 bytes
//                GLdouble padding1; //8 bytes
//                GLint padding2; //4 bytes

            public:
//                LineSegment(GLuint seedPointId, const glm::vec3& p1, const glm::vec3& p2)
//                    : seedPointId(seedPointId), //p1(p1, 1), p2(p2, 1), padding1(0), padding2(0) {}
//                      p1_x(p1.x), p1_y(p1.y), p1_z(p1.z),
//                      p2_x(p2.x), p2_y(p2.y), p2_z(p2.z)
//                {}
                LineSegment(GLuint seedPointId, const glm::vec3& p1, const glm::vec3& p2)
                        : seedPointId(seedPointId), p1(p1, 1), p2(p2, 1)
                {}

                double CalculateLength() const
                {
                    return glm::distance(
//                        glm::vec3(p1_x, p1_y, p1_z),
//                        glm::vec3(p2_x, p2_y, p2_z)
                        p1, p2
                    );
                }

                GLuint GetSeedPointId() const { return seedPointId; }
        };

        static unsigned int GLOBAL_FIBER_ID;

    private:
        const unsigned int id;
        const unsigned int seedPointId;

        std::vector<LineSegment> lineSegments;
        std::vector<glm::vec3> lineSegmentPoints;
        std::vector<glm::vec3> uniquePoints;

        
    public:
        explicit Fiber(unsigned int seedPointId);
        ~Fiber() = default;

        Fiber(const Fiber&) = delete;
        Fiber& operator=(const Fiber&) = delete;

        void AddSegment(const glm::vec3& p1, const glm::vec3& p2);
        const std::vector<LineSegment>& GetLineSegments() const;
        const std::vector<glm::vec3>& GetLineSegmentsAsPoints() const;
        const std::vector<glm::vec3>& GetUniquePoints() const;

        unsigned int GetId() const;
        unsigned int GetSeedPointId() const;

        //Test function, not performant
        double CalculateLength() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
