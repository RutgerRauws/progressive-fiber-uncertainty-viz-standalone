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
        class LineSegment
        {
            private:
                glm::vec4 p1;               // 16 bytes
                glm::vec4 p2;               // 16 bytes
                GLuint seedPointId;         // 4 bytes
                GLuint fiberId;             // 4 bytes
                GLuint padding2, padding3;  // 8 bytes

            public:
                LineSegment(GLuint seedPointId, GLuint fiberId, const glm::vec3& p1, const glm::vec3& p2)
                        : seedPointId(seedPointId), fiberId(fiberId), p1(p1, 1), p2(p2, 1)
                {}

                double CalculateLength() const
                {
                    return glm::distance(p1, p2);
                }

                GLuint GetSeedPointId() const { return seedPointId; }
                GLuint GetFiberId() const { return fiberId; }
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
