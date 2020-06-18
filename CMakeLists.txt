cmake_minimum_required(VERSION 3.9)
project(progressive_fiber_uncertainty_viz)

set(CMAKE_CXX_STANDARD 98)

set(VTK_DIR /home/rutger/Desktop/Studie/graduation/slicer/Slicer-SuperBuild-Debug/VTK-build)
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

add_executable(progressive_fiber_uncertainty_viz main.cpp)

target_link_libraries(progressive_fiber_uncertainty_viz ${VTK_LIBRARIES})

add_custom_command(TARGET progressive_fiber_uncertainty_viz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data/ $<TARGET_FILE_DIR:progressive_fiber_uncertainty_viz>/data/)