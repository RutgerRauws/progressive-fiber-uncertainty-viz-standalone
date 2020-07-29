cmake_minimum_required(VERSION 3.9)
project(progressive_fiber_uncertainty_viz)
set(CMAKE_CXX_STANDARD 98)

set(VTK_DIR /home/rutger/Desktop/Graduation/slicer/Slicer-SuperBuild-Debug/VTK-build)
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

add_executable(progressive_fiber_uncertainty_viz main.cpp KeyPressInteractorStyle.h FiberObserver.h FiberRenderer.cpp FiberRenderer.h FiberPublisher.cpp FiberPublisher.h Fiber.cpp Fiber.h Point.h VisitationMap.cpp VisitationMap.h Voxel.cpp Voxel.h VisitationMapUpdater.cpp VisitationMapUpdater.h VisitationMapDebugRenderer.cpp VisitationMapDebugRenderer.h VisitationMapRenderer.cpp VisitationMapRenderer.h IsovalueObserver.h)

target_link_libraries(progressive_fiber_uncertainty_viz ${VTK_LIBRARIES})

file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION data/)