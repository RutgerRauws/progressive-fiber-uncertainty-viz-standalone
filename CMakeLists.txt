cmake_minimum_required(VERSION 3.9)
project(progressive_fiber_uncertainty_viz)
set(CMAKE_CXX_STANDARD 98)

set(EXECUTABLE_NAME "progressive_fiber_uncertainty_viz")

set(VTK_DIR /home/rutger/Desktop/Graduation/slicer/Slicer-SuperBuild-Debug/VTK-build)
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

add_definitions(-DSTANDALONE)

add_executable(${EXECUTABLE_NAME}
        main.cpp
        src/interaction/InteractionManager.h
        src/util/FiberObserver.h
        src/util/FiberRenderer.cpp src/util/FiberRenderer.h
        src/util/FiberPublisher.cpp src/util/FiberPublisher.h
        src/util/Fiber.cpp src/util/Fiber.h
        src/interaction/KeyPressObserver.h
        src/centerline/CenterlineRenderer.cpp src/centerline/CenterlineRenderer.h
        src/centerline/DistanceTable.cpp src/centerline/DistanceTable.h
        src/visitationmap/VisitationMap.cpp src/visitationmap/VisitationMap.h
        src/visitationmap/VisitationMapUpdater.cpp src/visitationmap/VisitationMapUpdater.h
        src/visitationmap/VisitationMapRenderer.cpp src/visitationmap/VisitationMapRenderer.h
        src/util/Shader.cpp src/util/Shader.h
        src/util/ShaderProgram.cpp src/util/ShaderProgram.h
        src/interaction/MovementHandler.cpp src/interaction/MovementHandler.h
        )

#Find any version 2.X of SFML
#See the FindSFML.cmake file for additional details and instructions
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

## OpenGL
#find_package(OpenGL REQUIRED)
#include_directories(${OPENGL_INCLUDE_DIR})
#if (OPENGL_FOUND)
#    target_link_libraries(${EXECUTABLE_NAME} ${OPENGL_LIBRARIES})
#    # or: target_link_libraries(${EXECUTABLE_NAME} ${OPENGL_gl_LIBRARY})
#    target_link_libraries(${EXECUTABLE_NAME} m)  # if you use maths.h
#endif()

target_link_libraries(progressive_fiber_uncertainty_viz ${VTK_LIBRARIES} -lGL -lGLU -lGLEW sfml-graphics sfml-window sfml-system)

file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION data/)

set (source "${CMAKE_SOURCE_DIR}/src/visitationmap/gpu/shaders/")
set (destination "${CMAKE_CURRENT_BINARY_DIR}/shaders")
add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${source} ${destination}
        DEPENDS ${destination}
        COMMENT "symbolic link resources folder from ${source} => ${destination}"
)

