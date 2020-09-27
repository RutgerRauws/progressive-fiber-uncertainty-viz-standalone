//
// Created by rutger on 8/26/20.
//

#ifndef PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
#define PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H


#include <string>
#include <vector>
#include <vtkObject.h>

//temporary hardcoded input file
//const std::string INPUT_FILE_NAME = "./data/corpus-callosum.vtk";
//const std::string INPUT_FILE_NAME = "./data/fiber-samples-without-outliers.vtk";
//const std::string INPUT_FILE_NAME = "./data/fiber-samples-with-outliers.vtk";
const std::string INPUT_FILE_NAME = "./data/cst-1.vtk";
//const std::string INPUT_FILE_NAME = "./data/cst-20.vtk";
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

const unsigned int RENDER_INTERVAL_MS = 33; //30fps
const double CELL_SIZE = 1.0f;
const double SPLAT_KERNEL_RADIUS = 2.0f;

void render_callback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData);


#endif //PROGRESSIVEFIBERUNCERTAINTYVIZ_MAIN_H
