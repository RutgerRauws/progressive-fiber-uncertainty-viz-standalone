//
// Created by rutger on 7/2/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H

#include <vector>
#include "vtkSmartPointer.h"
#include "glm/ext.hpp"

class Fiber
{
    public:
        class LineSegment
        {
            private:
                glm::vec4 p1, p2; //it's a 4D-vector because this is required for padding in the shader

            public:
                LineSegment(glm::vec4& p1, glm::vec4& p2) : p1(p1), p2(p2) {}
                double CalculateLength() const { return glm::distance(p1, p2); }
        };

        static unsigned int GLOBAL_FIBER_ID;

    private:
        const unsigned int id;
        const unsigned int seedPointId;

        std::vector<LineSegment> lineSegments;
        std::vector<glm::vec4> lineSegmentPoints; //it's a 4D-vector because this is required for padding in the shader
        std::vector<glm::vec4> uniquePoints; //it's a 4D-vector because this is required for padding in the shader

        
    public:
        Fiber(unsigned int seedPointId);

        Fiber(const Fiber&) = delete;
        Fiber& operator=(const Fiber&) = delete;

        void AddSegment(const glm::vec3& p1, const glm::vec3& p2);
        const std::vector<LineSegment>& GetLineSegments() const;
        const std::vector<glm::vec4>& GetLineSegmentsAsPoints() const;
        const std::vector<glm::vec4>& GetUniquePoints() const;

        unsigned int GetId() const;
        unsigned int GetSeedPointId() const;

        //Test function, not performant
        double CalculateLength() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
