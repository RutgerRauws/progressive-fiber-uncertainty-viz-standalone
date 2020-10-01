//
// Created by rutger on 10/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADERTEST_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADERTEST_H

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

class ShaderTest
{
    private:
        const std::string VERTEX_SHADER_PATH   = "./shaders/vertex.glsl";
        const std::string FRAGMENT_SHADER_PATH = "./shaders/fragment.glsl";

        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkActor> actor;

        void initialize();

        static std::string ReadStringFromFile(const std::string& path);

    public:
        ShaderTest(vtkSmartPointer<vtkRenderer> renderer);
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADERTEST_H
