//
// Created by rutger on 8/26/20.
//

#ifndef PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
#define PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H

#include <string>
#include <vector>
#include <vtkObject.h>
#include "glm/vec3.hpp"

/*
 * Temporary hardcoded test input files
 */
//const std::vector<std::string> FIBER_FILE_NAMES = {
//    "./data/fibers/cc/cc-6/1.vtk",
//    "./data/fibers/cc/cc-6/2.vtk",
//    "./data/fibers/cc/cc-6/3.vtk",
//    "./data/fibers/cc/cc-6/4.vtk",
//    "./data/fibers/cc/cc-6/5.vtk",
//    "./data/fibers/cc/cc-6/6.vtk"
//};

const std::vector<std::string> FIBER_FILE_NAMES = {
        "./data/fibers/cst/cst-10/1.vtk",
        "./data/fibers/cst/cst-10/2.vtk",
        "./data/fibers/cst/cst-10/3.vtk",
        "./data/fibers/cst/cst-10/4.vtk",
        "./data/fibers/cst/cst-10/5.vtk",
        "./data/fibers/cst/cst-10/6.vtk",
        "./data/fibers/cst/cst-10/7.vtk",
        "./data/fibers/cst/cst-10/8.vtk",
        "./data/fibers/cst/cst-10/9.vtk",
        "./data/fibers/cst/cst-10/10.vtk",
};

const std::string DWI_PATH = "./data/mri/test-data-dwi-volume.nhdr";

#define DEBUG

#endif //PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
