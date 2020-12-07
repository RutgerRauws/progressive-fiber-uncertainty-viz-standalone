//
// Created by rutger on 11/18/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CONFIGURATION_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CONFIGURATION_H

#include <QtGui/QColor>

class Configuration
{
private:
    Configuration()
    {
        NUMBER_OF_REPRESENTATIVE_FIBERS = 25; //TODO: This is not being used in other classes yet! You need to set this yourself!
        SIDE_SIZE = 1.0f;

        SHOW_DWI_SLICES = true;
        SHOW_FIBER_SAMPLES = true;
        SHOW_REPRESENTATIVE_FIBERS = true;

        USE_TRILINEAR_INTERPOLATION = false;
        USE_FIBER_FREQUENCIES = true;

        ISOVALUE_MIN_FREQUENCY_PERCENTAGE = 0; //0%
        ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE = 1; //100%

        HULL_OPACITY = 1.0f; //fully opaque
        HULL_COLOR_AMBIENT  = QColor(200, 200, 0);
        HULL_COLOR_DIFFUSE  = QColor(200, 200, 0);
        HULL_COLOR_SPECULAR = QColor(50, 50, 50);
    };

public:
    //Setup
    unsigned int NUMBER_OF_REPRESENTATIVE_FIBERS;
    float SIDE_SIZE; //mm

    //Rendering
    bool SHOW_DWI_SLICES;
    bool SHOW_FIBER_SAMPLES;
    bool SHOW_REPRESENTATIVE_FIBERS;

    bool USE_TRILINEAR_INTERPOLATION;
    bool USE_FIBER_FREQUENCIES; //when false it will use distance scores

    float  ISOVALUE_MIN_FREQUENCY_PERCENTAGE;
    double ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE;

    float HULL_OPACITY;
//    float HULL_COLOR_R, HULL_COLOR_G, HULL_COLOR_B;
    QColor HULL_COLOR_AMBIENT, HULL_COLOR_DIFFUSE, HULL_COLOR_SPECULAR;

    static Configuration& getInstance()
    {
        static Configuration instance;
        return instance;
    }
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CONFIGURATION_H


