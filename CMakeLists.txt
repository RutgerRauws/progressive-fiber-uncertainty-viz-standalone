cmake_minimum_required(VERSION 3.9)
project(progressive_fiber_uncertainty_viz)
set(CMAKE_CXX_STANDARD 98)

set(VTK_DIR /home/rutger/Desktop/Graduation/slicer/Slicer-SuperBuild-Debug/VTK-build)
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

add_definitions(-DSTANDALONE)

add_executable(progressive_fiber_uncertainty_viz
        main.cpp
        src/interaction/KeyPressInteractorStyle.h
        src/util/FiberObserver.h
        src/util/FiberRenderer.cpp src/util/FiberRenderer.h
        src/util/FiberPublisher.cpp src/util/FiberPublisher.h
        src/util/Fiber.cpp src/util/Fiber.h
        src/util/Point.h
        src/visitationmap/VisitationMap.cpp src/visitationmap/VisitationMap.h
        src/visitationmap/Cell.cpp src/visitationmap/Cell.h
        src/visitationmap/VisitationMapUpdater.cpp src/visitationmap/VisitationMapUpdater.h
        src/visitationmap/VisitationMapDebugRenderer.cpp src/visitationmap/VisitationMapDebugRenderer.h
        src/visitationmap/VisitationMapRenderer.cpp src/visitationmap/VisitationMapRenderer.h
        src/interaction/KeyPressObserver.h
        src/centerline/CenterlineRenderer.cpp src/centerline/CenterlineRenderer.h
        src/centerline/DistanceTable.cpp src/centerline/DistanceTable.h
        src/visitationmap/gpu/ShaderTest.cpp src/visitationmap/gpu/ShaderTest.h
)

target_link_libraries(progressive_fiber_uncertainty_viz ${VTK_LIBRARIES})

file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION data/)

set (source "${CMAKE_SOURCE_DIR}/src/visitationmap/gpu/shaders/")
set (destination "${CMAKE_CURRENT_BINARY_DIR}/shaders")
add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${source} ${destination}
        DEPENDS ${destination}
        COMMENT "symbolic link resources folder from ${source} => ${destination}"
)

