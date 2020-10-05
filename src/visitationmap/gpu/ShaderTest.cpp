//
// Created by rutger on 10/1/20.
//

#include "ShaderTest.h"

#include <utility>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkShaderProperty.h>
#include <vtkCommand.h>
#include <vtkShaderProgram.h>
#include <vtkPolyData.h>
#include <vtkPolyLine.h>
#include <vtkCellData.h>
#include <vtkOpenGLUniforms.h>
#include <sstream>

#include <vtkOpenGLImageAlgorithmHelper.h>

//------------------------------------------------------------------------------
// Update a uniform in the shader for each render. We do this with a
// callback for the UpdateShaderEvent
class vtkShaderCallback : public vtkCommand
{
public:
    static vtkShaderCallback* New() { return new vtkShaderCallback; }

    vtkRenderer* Renderer;

    void Execute(vtkObject*, unsigned long, void* calldata) override
    {
        vtkShaderProgram* program = reinterpret_cast<vtkShaderProgram*>(calldata);

        float diffuseColor[3];

#if 0 // trippy mode
    float inputHSV[3];
    double theTime = vtkTimerLog::GetUniversalTime();
    double twopi = 2.0*vtkMath::Pi();

    inputHSV[0] = sin(twopi*fmod(theTime,3.0)/3.0)/4.0 + 0.25;
    inputHSV[1] = sin(twopi*fmod(theTime,4.0)/4.0)/2.0 + 0.5;
    inputHSV[2] = 0.7*(sin(twopi*fmod(theTime,19.0)/19.0)/2.0 + 0.5);
    vtkMath::HSVToRGB(inputHSV,diffuseColor);
    cellBO->Program->SetUniform3f("diffuseColorUniform", diffuseColor);

    if (this->Renderer)
    {
      inputHSV[0] = sin(twopi*fmod(theTime,5.0)/5.0)/4.0 + 0.75;
      inputHSV[1] = sin(twopi*fmod(theTime,7.0)/7.0)/2.0 + 0.5;
      inputHSV[2] = 0.5*(sin(twopi*fmod(theTime,17.0)/17.0)/2.0 + 0.5);
      vtkMath::HSVToRGB(inputHSV,diffuseColor);
      this->Renderer->SetBackground(diffuseColor[0], diffuseColor[1], diffuseColor[2]);

      inputHSV[0] = sin(twopi*fmod(theTime,11.0)/11.0)/2.0+0.5;
      inputHSV[1] = sin(twopi*fmod(theTime,13.0)/13.0)/2.0 + 0.5;
      inputHSV[2] = 0.5*(sin(twopi*fmod(theTime,17.0)/17.0)/2.0 + 0.5);
      vtkMath::HSVToRGB(inputHSV,diffuseColor);
      this->Renderer->SetBackground2(diffuseColor[0], diffuseColor[1], diffuseColor[2]);
    }
#else
        diffuseColor[0] = 0.4;
        diffuseColor[1] = 0.7;
        diffuseColor[2] = 0.6;
        program->SetUniform3f("diffuseColorUniform", diffuseColor);


//        std::vector<int> points(1024);
//        points[0] = 0;
//        points[1] = 1;
//        points[2] = 0;
//
//        program->SetAttributeArray(
//                "cells",
//                points,
//                3,
//                vtkShaderProgram::NormalizeOption::NoNormalize
//        );

#endif
    }

    vtkShaderCallback() { this->Renderer = nullptr; }
};



ShaderTest::ShaderTest(vtkSmartPointer<vtkRenderer> renderer)
    : renderer(std::move(renderer)),
      actor(vtkSmartPointer<vtkActor>::New())
{
    initialize();
}

void ShaderTest::initialize()
{
//    vtkNew<vtkOpenGLImageAlgorithmHelper> helper;
    /***
     *
     * Setting up OpenGL
     *
     */
    GLenum err = glewInit();
    if(err != GLEW_OK)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        throw std::runtime_error(reinterpret_cast<const char *>(glewGetErrorString(err)));
    }

    std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

    if(!GLEW_VERSION_4_3)
    {
        throw std::runtime_error("OpenGL version 4.3 is not supported.");
    }

    if(!GLEW_ARB_shader_storage_buffer_object)
    {
        /* Problem: we cannot use SSBOs, which is necessary to keep our algorithm performant. */
        throw std::runtime_error("SSBOs are not supported for this graphics card (missing ARB_shader_storage_buffer_object).");
    }

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    std::string source = readStringFromFile(COMPUTE_SHADER_PATH);
    const char* source_ptr = source.c_str();
    glShaderSource(shader,
                   1,
                   &source_ptr,
                   NULL
    );
    glCompileShader(shader);

    checkForErrors(shader); //Check for compilation errors

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    checkForErrors(shader); //Check for linking errors and validate program

    glUseProgram(program);

    /***
     *
     * Preparing inputs
     *
     */
    //Visitation Map Properties
    GLint vmProp_loc;
    vmProp_loc = glGetUniformLocation(program, "visitationMapProp.width");
    glProgramUniform1i(program, vmProp_loc, width);
    vmProp_loc = glGetUniformLocation(program, "visitationMapProp.height");
    glProgramUniform1i(program, vmProp_loc, height);
    vmProp_loc = glGetUniformLocation(program, "visitationMapProp.depth");
    glProgramUniform1i(program, vmProp_loc, depth);
    vmProp_loc = glGetUniformLocation(program, "visitationMapProp.cellSize");
    glProgramUniform1f(program, vmProp_loc, spacing);

    //Visitation Map frequencies itself
    unsigned int frequency_data[width * height * depth];
    std::fill_n(frequency_data, width * height * depth, 0);

    GLuint frequency_map_ssbo;
    glGenBuffers(1, &frequency_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int) * width * height * depth, &frequency_data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frequency_map_ssbo);
//    glDispatchCompute(1, 1, 1); //(nr-of-segments / local-group-size, 1, 1)
    glDispatchCompute(10, 1, 1);

    //Sync here to make writes visible
    //glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    //glMemoryBarrier(GL_COMPUTE_SHADER_BIT);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /***
     *
     * Preparing outputs
     *
     */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, frequency_map_ssbo);
    GLuint* ptr = (GLuint*) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    memcpy(frequency_data, ptr, sizeof(frequency_data));

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    unsigned int count = 0;
    for(unsigned int i = 0; i < width * height * depth; i++)
    {
        if(frequency_data[i] > 1) {
            std::cout << frequency_data[i] << std::endl;
            count++;
        }
    }

//    std::cout << "Counted " << count << " items of the " << width * height * depth << " total." << std::endl;


    /***
     *
     * Unrelated stuff
     *
     */
    vtkNew<vtkOpenGLPolyDataMapper> mapper;

    renderer->AddActor(actor.Get());
    renderer->GradientBackgroundOn();

    vtkNew<vtkPolyLine> line;
    line->GetPoints()->InsertPoint(0, 1, 0, 0);
    line->GetPoints()->InsertPoint(1, 1, 1, 0);
    line->GetPoints()->InsertPoint(2, 0, 0, 1);
    line->GetPoints()->InsertPoint(3, 0, 1, 0);
    line->GetPoints()->InsertPoint(4, 1, 0, 1);

    line->GetPointIds()->SetNumberOfIds(5);
    line->GetPointIds()->SetId(0, 0);
    line->GetPointIds()->SetId(1, 1);
    line->GetPointIds()->SetId(2, 2);
    line->GetPointIds()->SetId(3, 3);
    line->GetPointIds()->SetId(4, 4);

    vtkNew<vtkCellArray> polyLines;
    polyLines->InsertNextCell(line);

    vtkNew<vtkPolyData> polyData;
    polyData->SetPoints(line->GetPoints());
    polyData->SetLines(polyLines);

//    vtkNew<vtkPointData> pointData;
//    polyData->SetPoints(vtkSmartPointer<vtkPoints>::New());
//    polyData->GetPoints()->InsertPoint(0, 1, 0, 0);
//    polyData->GetPoints()->InsertPoint(1, 1, 1, 0);
//    polyData->GetPoints()->InsertPoint(2, 0, 0, 1);
//    polyData->GetPoints()->InsertPoint(3, 0, 1, 0);
//    polyData->GetPoints()->InsertPoint(4, 1, 0, 1);
//
//    mapper->SetInputData(pointData);

//    mapper->SetInputConnection(polyData)
    mapper->SetInputData(polyData);

//    vtkNew<vtkPolyDataNormals> norms;
//    norms->SetInputConnection(reader->GetOutputPort());
//    norms->Update();
//
//    mapper->SetInputConnection(norms->GetOutputPort());

    actor->SetMapper(mapper.Get());
    actor->GetProperty()->SetAmbientColor(0.2, 0.2, 1.0);
    actor->GetProperty()->SetDiffuseColor(1.0, 0.65, 0.7);
    actor->GetProperty()->SetSpecularColor(1.0, 1.0, 1.0);
    actor->GetProperty()->SetSpecular(0.5);
    actor->GetProperty()->SetDiffuse(0.7);
    actor->GetProperty()->SetAmbient(0.5);
    actor->GetProperty()->SetSpecularPower(20.0);
    actor->GetProperty()->SetOpacity(1.0);

    vtkShaderProperty* sp = actor->GetShaderProperty();

    // Clear all custom shader tag replacements
    sp->ClearAllVertexShaderReplacements();
    sp->ClearAllFragmentShaderReplacements();
    sp->ClearAllGeometryShaderReplacements();
    sp->ClearAllShaderReplacements();


//    sp->SetVertexShaderCode(
//        readStringFromFile(VERTEX_SHADER_PATH).frequency_data()
//    );
//
//    sp->SetGeometryShaderCode(
//        readStringFromFile(GEOMETRY_SHADER_PATH).frequency_data()
//    );
//
//    sp->SetFragmentShaderCode(
//        readStringFromFile(FRAGMENT_SHADER_PATH).frequency_data()
//    );


//    std::vector<int> points(1024);
//    points[0] = 0;
//    points[1] = 1;
//    points[2] = 0;
//
//    sp->GetFragmentCustomUniforms()->SetUniform(
//        "cells",
//        vtkUniforms::TupleType::TupleTypeVector,
//        points.capacity(),
//        points
//    );

    // Setup a callback to change some uniforms
    vtkNew<vtkShaderCallback> myCallback;
    myCallback->Renderer = renderer;
    mapper->AddObserver(vtkCommand::UpdateShaderEvent, myCallback);

}

std::string ShaderTest::readStringFromFile(const std::string& path)
{
    std::ifstream t(path);
    std::stringstream buffer;
    buffer << t.rdbuf();

    return buffer.str();
}

void ShaderTest::checkForErrors(GLuint shader)
{
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if(success == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        GLchar errorLog[maxLength];
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

        // Provide the infolog in whatever manor you deem best.
        // Exit with failure.
        glDeleteShader(shader); // Don't leak the shader.

        throw std::runtime_error(std::string("Shader failed to compile/link:\n") + std::string(errorLog));
    }
}
