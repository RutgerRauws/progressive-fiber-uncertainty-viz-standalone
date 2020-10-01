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
#include <sstream>

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
    // The following code is mainly for regression testing as we do not have any
    // custom shader replacements.
    sp->ClearAllVertexShaderReplacements();
    sp->ClearAllFragmentShaderReplacements();
    sp->ClearAllGeometryShaderReplacements();
    sp->ClearAllShaderReplacements();


    sp->SetVertexShaderCode(
        ReadStringFromFile(VERTEX_SHADER_PATH).data()
    );

    sp->SetFragmentShaderCode(
        ReadStringFromFile(FRAGMENT_SHADER_PATH).data()
    );

    // Setup a callback to change some uniforms
    vtkNew<vtkShaderCallback> myCallback;
    myCallback->Renderer = renderer;
    mapper->AddObserver(vtkCommand::UpdateShaderEvent, myCallback);

}

std::string ShaderTest::ReadStringFromFile(const std::string& path)
{
    std::ifstream t(path);
    std::stringstream buffer;
    buffer << t.rdbuf();

    return buffer.str();
}
