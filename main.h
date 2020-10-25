//
// Created by rutger on 8/26/20.
//

#ifndef PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
#define PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H

#include <string>
#include <vector>
#include <vtkObject.h>
#include "src/util/glm/vec3.hpp"

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

/*
 * RENDERING
 */
const unsigned int SCREEN_WIDTH  = 900; //pixels
const unsigned int SCREEN_HEIGHT = 700; //pixels
const unsigned int RENDER_INTERVAL_MS = 33; //30fps

const glm::vec3 CAMERA_POS(367.59, 197.453, 328.134);
const glm::vec3 CAMERA_FRT(-0.678897, -0.406737, -0.611281);
const glm::vec3 CAMERA_UP(0, 1, 0);

#define DEBUG

#endif //PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
