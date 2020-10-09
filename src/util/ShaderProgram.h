#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADER_PROGRAM_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADER_PROGRAM_H

#include "Shader.h"

class ShaderProgram
{
    private:
        GLuint id;

    public:
        ShaderProgram (Shader** shaders, GLuint num);
        ~ShaderProgram();

        void Use();
        GLuint GetId ();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SHADER_PROGRAM_H
