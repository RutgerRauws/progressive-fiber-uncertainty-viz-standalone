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
const std::string  FIBER_FOLDER_PATH = "./data/open-source/fibers/cst-10/";
const std::string  MRI_FILE_PATH     = "./data/open-source/dwi/test-data-dwi-volume.nhdr";
const unsigned int MRI_DIMENSION_X   = 112;
const unsigned int MRI_DIMENSION_Y   = 112;
const unsigned int MRI_DIMENSION_Z   = 70;
const float        MRI_VOXEL_SIZE    = 2; //mm


#define DEBUG

#endif //PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
