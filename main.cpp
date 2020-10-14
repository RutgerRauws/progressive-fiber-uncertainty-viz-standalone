#include "main.h"

#include <GL/glew.h>
//#include <iostream>
//#include <X11/Xlib.h>

#include "src/interaction/InteractionManager.h"
#include "src/util/FiberPublisher.h"
#include "src/util/FiberRenderer.h"
#include "src/centerline/CenterlineRenderer.h"
#include "src/visitationmap/VisitationMapUpdater.h"
#include "src/util/ShaderProgram.h"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "src/util/glm/glm.hpp"
#include "src/util/glm/gtc/matrix_transform.hpp"
#include "src/util/glm/gtc/type_ptr.hpp"
#include "src/interaction/MovementHandler.h"
#include "src/visitationmap/VisitationMapRenderer.h"

int main()
{
    std::cout << "Application started..." << std::endl;

    FiberPublisher fiberPublisher(INPUT_FILE_NAME);
//    FiberPublisher fiberPublisher(INPUT_FILE_NAMES);

    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antialiasingLevel = 2; // Optional

    // Request OpenGL version 4.3
    settings.majorVersion = 4;
    settings.minorVersion = 3;
    settings.attributeFlags = sf::ContextSettings::Core;

    sf::RenderWindow window(
        sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT),
        "Progressive Fiber Uncertainty Visualization",
        sf::Style::Close,
        settings
    );

    window.setFramerateLimit(60);

    glewExperimental = GL_TRUE;
    glewInit();
    window.display();
    Shader* visitationMapVS = nullptr;
    Shader* visitationMapFS = nullptr;

    Shader* fibersVS = nullptr;
    Shader* fibersFS = nullptr;

    try
    {
        visitationMapVS = Shader::LoadFromFile(VERTEX_SHADER_VM_PATH, GL_VERTEX_SHADER);
        visitationMapVS->Compile();

        fibersVS = Shader::LoadFromFile(VERTEX_SHADER_FIB_PATH, GL_VERTEX_SHADER);
        fibersVS->Compile();
    }
    catch(const ShaderError& e)
    {
        std::cerr << "Could not compile vertex shader: " << e.what() << std::endl;
        return -1;
    }

    try
    {
        visitationMapFS = Shader::LoadFromFile(FRAGMENT_SHADER_VM_PATH, GL_FRAGMENT_SHADER);
        visitationMapFS->Compile();

        fibersFS = Shader::LoadFromFile(FRAGMENT_SHADER_FIB_PATH, GL_FRAGMENT_SHADER);
        fibersFS->Compile();
    }
    catch(const ShaderError& e)
    {
        std::cerr << "Could not compile fragment shader: " << e.what() << std::endl;
        return -1;
    }

    Shader* shadersVM[2] = {visitationMapVS, visitationMapFS};
    ShaderProgram visitationMapShaderProgram(shadersVM, 2);
//    visitationMapShaderProgram.Use();

    Shader* shadersFib[2] = {fibersVS, fibersFS};
    ShaderProgram fibersShaderProgram(shadersFib, 2);
//    fibersShaderProgram.Use();

    glm::mat4 modelMat = glm::mat4(1.0f);
//    modelMat = glm::rotate(modelMat, glm::radians(-55.0f), glm::vec3(1, 0, 0));

    glm::mat4 viewMat = glm::mat4(1.0f);
    viewMat = glm::translate(viewMat, glm::vec3(0, 0, -3));

    glm::mat4 projMat;
    projMat = glm::perspective(
            glm::radians(45.0f),
            (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT,
            0.1f,
            10000.0f
    );

//    Box box(-0.5,0.5,-0.5,0.5,-0.5,0.5);
//    Box box(fiberPublisher.GetBounds());
    VisitationMapRenderer box(DTI_XMIN, DTI_XMAX, DTI_YMIN, DTI_YMAX, DTI_ZMIN, DTI_ZMAX);
    FiberRenderer fiberRenderer;

    fiberPublisher.RegisterObserver(fiberRenderer);

//
//                                    GLuint vao;
//                                    glGenVertexArrays(1, &vao);
//                                    glBindVertexArray(vao);
//
//                                    GLuint vertexBuffer;
//                                    glGenBuffers(1, &vertexBuffer);
//                                    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
//                                    glBufferData(GL_ARRAY_BUFFER, box.GetNumberOfBytes(), box.GetVertexBufferData(), GL_STATIC_DRAW);

//    GLuint elementBuffer;
//    glGenBuffers(1, &elementBuffer);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffer);
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

//                                    GLint posAttrib = glGetAttribLocation(shaderProgram.GetId(), "position");
//                                    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
//                                    glEnableVertexAttribArray(posAttrib);

    fibersShaderProgram.Use();
    int modelMatFibLoc = glGetUniformLocation(visitationMapShaderProgram.GetId(), "modelMat");
    int viewMatFibLoc = glGetUniformLocation(visitationMapShaderProgram.GetId(), "viewMat");
    int projMatFibLoc = glGetUniformLocation(visitationMapShaderProgram.GetId(), "projMat");

    visitationMapShaderProgram.Use();
    int modelMatVMLoc = glGetUniformLocation(visitationMapShaderProgram.GetId(), "modelMat");
    int viewMatVMLoc = glGetUniformLocation(visitationMapShaderProgram.GetId(), "viewMat");
    int projMatVMLoc = glGetUniformLocation(visitationMapShaderProgram.GetId(), "projMat");

    box.SetUpUniforms(visitationMapShaderProgram.GetId());

    InteractionManager interactionManager;
    MovementHandler movementHandler(window, modelMat, viewMat, projMat, visitationMapShaderProgram.GetId());
    movementHandler.SetCameraPosition(CAMERA_POS);
    movementHandler.SetCameraFront(CAMERA_FRT);

    glEnable(GL_DEPTH_TEST);

    fiberPublisher.Start();

    while (window.isOpen())
    {
        movementHandler.update();

        sf::Event windowEvent;
        while (window.pollEvent(windowEvent))
        {
            interactionManager.HandleInteraction(windowEvent);

            switch (windowEvent.type)
            {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::KeyPressed:
                    //window.close();
                    break;
                default:
                    break;
            }
        }

        // clear the window with black color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //Actual draw calls

        visitationMapShaderProgram.Use();
        glUniformMatrix4fv(modelMatVMLoc, 1, GL_FALSE, glm::value_ptr(modelMat));
        glUniformMatrix4fv(viewMatVMLoc, 1, GL_FALSE, glm::value_ptr(viewMat));
        glUniformMatrix4fv(projMatVMLoc, 1, GL_FALSE, glm::value_ptr(projMat));
        box.Render();


        fibersShaderProgram.Use();
        glUniformMatrix4fv(modelMatFibLoc, 1, GL_FALSE, glm::value_ptr(modelMat));
        glUniformMatrix4fv(viewMatFibLoc, 1, GL_FALSE, glm::value_ptr(viewMat));
        glUniformMatrix4fv(projMatFibLoc, 1, GL_FALSE, glm::value_ptr(projMat));
        fiberRenderer.Render();




        window.display();
    }

    delete visitationMapVS;
    delete visitationMapFS;

    return 0;
}

//int main()
//{
//    XInitThreads();
//
//    std::cout << "Application started." << std::endl;
//
//
//
//
//    /*
//     * Starting main functionality
//     */
//    FiberPublisher fiberPublisher(INPUT_FILE_NAME);
////    FiberPublisher fiberPublisher(INPUT_FILE_NAMES);
//
//    CenterlineRenderer centerlineRenderer(renderer);
//    FiberRenderer fiberRenderer(renderer);
//
//    VisitationMapUpdater shaderTest(renderer, fiberPublisher.GetBounds(), 2.0f);
//
//    keypressHandler->AddObserver("f", &fiberRenderer); //Toggle rendering of fibers.
//    keypressHandler->AddObserver("p", &fiberRenderer); //Toggle rendering of points of fibers.
//    keypressHandler->AddObserver("c", &centerlineRenderer); //Toggle rendering of centerline.
//
//    fiberPublisher.RegisterObserver(fiberRenderer);
//    fiberPublisher.RegisterObserver(centerlineRenderer);
//    fiberPublisher.Start();
//
//    //renderWindowInteractor->Initialize();
//    renderWindowInteractor->Start();
//
//    //When we reach this point, the renderWindowInteractor has been terminated by the KeyPressInteractorStyle
//    fiberPublisher.Stop();
//
//    return EXIT_SUCCESS;
//}