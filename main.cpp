#include "main.h"

#include <GL/glew.h>

#include "src/interaction/InteractionManager.h"
#include "src/util/FiberPublisher.h"
#include "src/util/FiberRenderer.h"
#include "src/centerline/CenterlineRenderer.h"
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "src/util/glm/glm.hpp"
#include "src/interaction/MovementHandler.h"
#include "src/visitationmap/VisitationMapRenderer.h"
#include "src/visitationmap/VisitationMapUpdater.h"
#include "src/interaction/WindowHandler.h"


void GLAPIENTRY
MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
    fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
             ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
             type, severity, message );
}

int main()
{
    /*
     * Context and window setup
     */
    std::cout << "Application started..." << std::endl;

//    FiberPublisher fiberPublisher(INPUT_FILE_NAME);
    FiberPublisher fiberPublisher(INPUT_FILE_NAMES);

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

    #ifdef DEBUG
    // During init, enable debug output
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(MessageCallback, 0);
    #endif

    /*
     * View setup
     */
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

    WindowHandler windowHandler(window);
    InteractionManager interactionManager;
    interactionManager.AddObserver(sf::Keyboard::Q, &windowHandler);

    MovementHandler movementHandler(window, modelMat, viewMat, projMat);
    movementHandler.SetCameraPosition(CAMERA_POS);
    movementHandler.SetCameraFront(CAMERA_FRT);

    /*
     * Visitation map handling
     */
    std::cout << "Initializing visitation map... " << std::flush;
    VisitationMap visitationMap = VisitationMap::CreateTest();
    std::cout << "Complete." << std::endl;

    RegionsOfInterest regionsOfInterest(fiberPublisher.GetNumberOfSeedPoints());

    VisitationMapUpdater visitationMapUpdater(visitationMap, regionsOfInterest);
    fiberPublisher.RegisterObserver(visitationMapUpdater);

    VisitationMapRenderer visitationMapRenderer(visitationMap, regionsOfInterest, movementHandler.GetCameraState());
    fiberPublisher.RegisterObserver(visitationMapRenderer);
    interactionManager.AddObserver(sf::Keyboard::U, &visitationMapRenderer);
    interactionManager.AddObserver(sf::Keyboard::J, &visitationMapRenderer);

    FiberRenderer fiberRenderer(movementHandler.GetCameraState());
    fiberPublisher.RegisterObserver(fiberRenderer);
    interactionManager.AddObserver(sf::Keyboard::F, &fiberRenderer);

    /*
     * Distance score calculations
     */
    DistanceTablesUpdater distanceTablesUpdater(fiberPublisher.GetNumberOfSeedPoints());
    fiberPublisher.RegisterObserver(distanceTablesUpdater);

    CenterlineRenderer centerlineRenderer(distanceTablesUpdater.GetDistanceTables(), movementHandler.GetCameraState());
    fiberPublisher.RegisterObserver(centerlineRenderer);
    interactionManager.AddObserver(sf::Keyboard::C, &centerlineRenderer);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    /*
     * Start render loop
     */
    window.display();
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
                default:
                    break;
            }
        }

        // clear the window with black color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //Actual draw calls
        visitationMapUpdater.Update();
        visitationMapRenderer.Render();
        centerlineRenderer.Render();
        fiberRenderer.Render();

        window.display();
    }

    fiberPublisher.Stop();

    return 0;
}