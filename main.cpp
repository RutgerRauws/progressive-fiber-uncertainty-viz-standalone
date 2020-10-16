#include "main.h"

#include <GL/glew.h>

#include "src/interaction/InteractionManager.h"
#include "src/util/FiberPublisher.h"
#include "src/util/FiberRenderer.h"
#include "src/centerline/CenterlineRenderer.h"
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "src/util/glm/glm.hpp"
#include "src/util/glm/gtc/matrix_transform.hpp"
#include "src/interaction/MovementHandler.h"
#include "src/visitationmap/VisitationMapRenderer.h"
#include "src/visitationmap/VisitationMapUpdater.h"

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

    window.display();

    glm::mat4 modelMat = glm::mat4(1.0f);

    glm::mat4 viewMat = glm::mat4(1.0f);
    viewMat = glm::translate(viewMat, glm::vec3(0, 0, -3));

    glm::mat4 projMat;
    projMat = glm::perspective(
            glm::radians(45.0f),
            (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT,
            0.1f,
            10000.0f
    );

    InteractionManager interactionManager;
    MovementHandler movementHandler(window, modelMat, viewMat, projMat);
    movementHandler.SetCameraPosition(CAMERA_POS);
    movementHandler.SetCameraFront(CAMERA_FRT);

    VisitationMap visitationMap = VisitationMap::CreateTest();

    VisitationMapUpdater visitationMapUpdater(visitationMap);
    fiberPublisher.RegisterObserver(visitationMapUpdater);

    VisitationMapRenderer visitationMapRenderer(visitationMap, movementHandler.GetCameraState());

    FiberRenderer fiberRenderer(movementHandler.GetCameraState());
    fiberPublisher.RegisterObserver(fiberRenderer);

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
        visitationMapRenderer.Render();
        fiberRenderer.Render();

        window.display();
    }

    return 0;
}