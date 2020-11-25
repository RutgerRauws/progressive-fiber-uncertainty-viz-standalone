cmake_minimum_required(VERSION 3.9)
project(progressive_fiber_uncertainty_viz)
set(CMAKE_CXX_STANDARD 98)

set(EXECUTABLE_NAME "progressive_fiber_uncertainty_viz")

set(VTK_DIR /home/rutger/Desktop/Graduation/slicer/Slicer-SuperBuild-Debug/VTK-build)
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

# For loading DWI data
set(ITK_DIR /home/rutger/Desktop/Graduation/slicer/Slicer-SuperBuild-Debug/ITK-build)
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

add_definitions(-DSTANDALONE)

find_package(PNGwriter 0.7.0)

## Qt5
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
find_package(Qt5Widgets REQUIRED)

add_executable(${EXECUTABLE_NAME}
    main.cpp
#    src/interaction/InteractionManager.h
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
    src/interaction/MovementHandler.h
    src/interaction/FPSMovementHandler.cpp src/interaction/FPSMovementHandler.h
    src/interaction/TrackBallMovementHandler.cpp src/interaction/TrackBallMovementHandler.h
    src/centerline/DistanceTablesUpdater.cpp src/centerline/DistanceTablesUpdater.h
    src/centerline/DistanceTableCollection.cpp src/centerline/DistanceTableCollection.h
    src/gui/UserInterface.cpp src/gui/UserInterface.h
    src/gui/OGLWidget.cpp src/gui/OGLWidget.h
    src/util/Camera.cpp src/util/Camera.h
    Configuration.h
    src/util/GL.h
    src/dwi/DWIDataReader.cpp src/dwi/DWIDataReader.h
    src/dwi/DWIRenderer.cpp src/dwi/DWIRenderer.h
    src/dwi/DWISlice.h
    src/gui/SelectColorButton.cpp src/gui/SelectColorButton.h
)

INCLUDE_DIRECTORIES("libs")

target_link_libraries(progressive_fiber_uncertainty_viz
    ${VTK_LIBRARIES}
    ${ITK_LIBRARIES}
    -lGL
    Qt5::Core Qt5::Gui Qt5::Widgets
    PNGwriter::PNGwriter
)

file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION data/)

set (source "${CMAKE_SOURCE_DIR}/src/shaders/")
set (destination "${CMAKE_CURRENT_BINARY_DIR}/shaders")
add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${source} ${destination}
        DEPENDS ${destination}
        COMMENT "symbolic link resources folder from ${source} => ${destination}"
)

