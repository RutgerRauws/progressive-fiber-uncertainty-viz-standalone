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
const float        MRI_NORMALIZATION = 20;

//const std::string  FIBER_FOLDER_PATH = "./data/closed-source/fibers/cst-40-regular/";
//const std::string  MRI_FILE_PATH     = "./data/closed-source/t1/t1_regular.nii";
//const unsigned int MRI_DIMENSION_X   = 112;
//const unsigned int MRI_DIMENSION_Y   = 112;
//const unsigned int MRI_DIMENSION_Z   = 70;
//const float        MRI_VOXEL_SIZE    = 2; //mm
//const float        MRI_NORMALIZATION = 10;

//Not properly rotated?
//const std::string  FIBER_FOLDER_PATH = "./data/closed-source/fibers/cst-68-tumor/";
//const std::string  MRI_FILE_PATH     = "./data/closed-source/t1/t1_tumor.nii";
//const unsigned int MRI_DIMENSION_X   = 112;
//const unsigned int MRI_DIMENSION_Y   = 112;
//const unsigned int MRI_DIMENSION_Z   = 70;
//const float        MRI_VOXEL_SIZE    = 2; //mm
//const float        MRI_NORMALIZATION = 10;

//const std::string  FIBER_FOLDER_PATH = "./data/closed-source/fibers/af-29-regular/";
//const std::string  MRI_FILE_PATH     = "./data/closed-source/t1/t1_regular.nii";
//const unsigned int MRI_DIMENSION_X   = 112;
//const unsigned int MRI_DIMENSION_Y   = 112;
//const unsigned int MRI_DIMENSION_Z   = 70;
//const float        MRI_VOXEL_SIZE    = 2; //mm
//const float        MRI_NORMALIZATION = 10;


#define DEBUG

#endif //PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
