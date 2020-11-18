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
//const std::string INPUT_FILE_NAME = "./data/corpus-callosum.vtk";
//const std::string INPUT_FILE_NAME = "./data/fiber-samples-without-outliers.vtk";
//const std::string INPUT_FILE_NAME = "./data/fiber-samples-with-outliers.vtk";
//const std::string INPUT_FILE_NAME = "./data/cst-1.vtk";
const std::string INPUT_FILE_NAME = "./data/cst-20.vtk";
//const std::string INPUT_FILE_NAME = "./data/slf-1.vtk";
//const std::string INPUT_FILE_NAME = "./data/slf-20.vtk";
//const std::string INPUT_FILE_NAME = "./data/cc-5.vtk";
//const std::string INPUT_FILE_NAME = "./data/for-each-seedpoint/1.vtk";

//const std::vector<std::string> INPUT_FILE_NAMES = {
//    "./data/cc-6/1.vtk",
//    "./data/cc-6/2.vtk",
//    "./data/cc-6/3.vtk",
//    "./data/cc-6/4.vtk",
//    "./data/cc-6/5.vtk",
//    "./data/cc-6/6.vtk"
//};

const std::vector<std::string> INPUT_FILE_NAMES = {
        "./data/cst-10/1.vtk",
        "./data/cst-10/2.vtk",
        "./data/cst-10/3.vtk",
        "./data/cst-10/4.vtk",
        "./data/cst-10/5.vtk",
        "./data/cst-10/6.vtk",
        "./data/cst-10/7.vtk",
        "./data/cst-10/8.vtk",
        "./data/cst-10/9.vtk",
        "./data/cst-10/10.vtk",
};

#define DEBUG

#endif //PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
