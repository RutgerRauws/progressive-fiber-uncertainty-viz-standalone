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

const std::vector<std::string> INPUT_FILE_NAMES = {
    "./data/for-each-seedpoint/1.vtk",
    "./data/for-each-seedpoint/2.vtk",
    "./data/for-each-seedpoint/3.vtk",
    "./data/for-each-seedpoint/4.vtk",
    "./data/for-each-seedpoint/5.vtk",
    "./data/for-each-seedpoint/6.vtk"
};

const float DTI_XMIN = -112;
const float DTI_XMAX =  112;
const float DTI_YMIN = -112;
const float DTI_YMAX =  112;
const float DTI_ZMIN = -70;
const float DTI_ZMAX =  70;

/*
 * RENDERING
 */
const unsigned int SCREEN_WIDTH  = 900; //pixels
const unsigned int SCREEN_HEIGHT = 700; //pixels
const unsigned int RENDER_INTERVAL_MS = 33; //30fps

const std::string VERTEX_SHADER_VM_PATH   = "./shaders/visitationmap/vertex.glsl";
const std::string FRAGMENT_SHADER_VM_PATH = "./shaders/visitationmap/fragment.glsl";

const std::string VERTEX_SHADER_FIB_PATH   = "./shaders/fibers/vertex.glsl";
const std::string FRAGMENT_SHADER_FIB_PATH = "./shaders/fibers/fragment.glsl";

const glm::vec3 CAMERA_POS(367.59, 197.453, 328.134);
const glm::vec3 CAMERA_FRT(-0.678897, -0.406737, -0.611281);
const glm::vec3 CAMERA_UP(0, 1, 0);

#endif //PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
