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
        static unsigned int GLOBAL_FIBER_ID;

    private:
        const unsigned int id;
        const unsigned int seedPointId;
        std::vector<glm::vec4> points; //it's a 4D-vector because this is required for padding in the shader
        
    public:
        Fiber(unsigned int seedPointId);

        Fiber(const Fiber&) = delete;
        Fiber& operator=(const Fiber&) = delete;

        void AddPoint(double x, double y, double z);
        const std::vector<glm::vec4>& GetPoints() const;

        unsigned int GetId() const;
        unsigned int GetSeedPointId() const;

        double CalculateLength() const;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_FIBER_H
