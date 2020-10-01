cmake_minimum_required(VERSION 3.9)
project(progressive_fiber_uncertainty_viz)
set(CMAKE_CXX_STANDARD 98)

set(VTK_DIR /home/rutger/Desktop/Graduation/slicer/Slicer-SuperBuild-Debug/VTK-build)
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

add_definitions(-DSTANDALONE)

add_executable(progressive_fiber_uncertainty_viz main.cpp KeyPressInteractorStyle.h FiberObserver.h FiberRenderer.cpp FiberRenderer.h FiberPublisher.cpp FiberPublisher.h Fiber.cpp Fiber.h Point.h VisitationMap.cpp VisitationMap.h Cell.cpp Cell.h VisitationMapUpdater.cpp VisitationMapUpdater.h VisitationMapDebugRenderer.cpp VisitationMapDebugRenderer.h VisitationMapRenderer.cpp VisitationMapRenderer.h KeyPressObserver.h CenterlineRenderer.cpp CenterlineRenderer.h DistanceTable.cpp DistanceTable.h gpu/ShaderTest.cpp gpu/ShaderTest.h)

target_link_libraries(progressive_fiber_uncertainty_viz ${VTK_LIBRARIES})

file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION data/)

set (source "${CMAKE_SOURCE_DIR}/gpu/shaders/")
set (destination "${CMAKE_CURRENT_BINARY_DIR}/shaders")
add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${source} ${destination}
        DEPENDS ${destination}
        COMMENT "symbolic link resources folder from ${source} => ${destination}"
)

