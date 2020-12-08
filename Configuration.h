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

        SHOW_AXIAL_PLANE = false;
        SHOW_CORONAL_PLANE = false;
        SHOW_SAGITTAL_PLANE = false;

        SHOW_FIBER_SAMPLES = true;
        SHOW_REPRESENTATIVE_FIBERS = true;

        USE_TRILINEAR_INTERPOLATION = false;
        USE_FIBER_FREQUENCIES = true;

        HULL_ISOVALUE_MIN_FREQUENCY_PERCENTAGE = 0.0f; //0%
        HULL_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE = 1.0f; //100%

        HULL_OPACITY = 1.0f; //fully opaque
        HULL_COLOR_AMBIENT  = QColor(200, 200, 0);
        HULL_COLOR_DIFFUSE  = QColor(200, 200, 0);
        HULL_COLOR_SPECULAR = QColor(50, 50, 50);

        SILHOUETTE_ISOVALUE_MIN_FREQUENCY_PERCENTAGE = 0.0f; //0%
        SILHOUETTE_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE = 1.0f; //100%

        SILHOUETTE_OPACITY = 0.6f;
        SILHOUETTE_COLOR   = QColor(0, 0, 255);
    };

public:
    //Setup
    unsigned int NUMBER_OF_REPRESENTATIVE_FIBERS;
    float SIDE_SIZE; //mm

    //Rendering
    bool SHOW_AXIAL_PLANE;
    bool SHOW_CORONAL_PLANE;
    bool SHOW_SAGITTAL_PLANE;

    bool SHOW_FIBER_SAMPLES;
    bool SHOW_REPRESENTATIVE_FIBERS;

    bool USE_TRILINEAR_INTERPOLATION;
    bool USE_FIBER_FREQUENCIES; //when false it will use distance scores

    float  HULL_ISOVALUE_MIN_FREQUENCY_PERCENTAGE;
    double HULL_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE;

    float HULL_OPACITY;
    QColor HULL_COLOR_AMBIENT, HULL_COLOR_DIFFUSE, HULL_COLOR_SPECULAR;

    float  SILHOUETTE_ISOVALUE_MIN_FREQUENCY_PERCENTAGE;
    double SILHOUETTE_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE;

    float SILHOUETTE_OPACITY;
    QColor SILHOUETTE_COLOR;

    static Configuration& getInstance()
    {
        static Configuration instance;
        return instance;
    }
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CONFIGURATION_H


