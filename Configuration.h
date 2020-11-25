//
// Created by rutger on 11/18/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CONFIGURATION_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CONFIGURATION_H

class Configuration
{
private:
    Configuration()
    {
        NUMBER_OF_REPRESENTATIVE_FIBERS = 25; //TODO: This is not being used in other classes yet! You need to set this yourself!
        SIDE_SIZE = 1.0f;

        SHOW_FIBER_SAMPLES = true;
        SHOW_REPRESENTATIVE_FIBERS = true;

        USE_TRILINEAR_INTERPOLATION = false;
        USE_FIBER_FREQUENCIES = true;

        ISOVALUE_MIN_FREQUENCY_PERCENTAGE = 0; //0%
        ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE = 1; //100%

        HULL_OPACITY = 1.0f; //fully opaque
        DWI_OPACITY  = 0.75f;
    };

public:
    //Setup
    unsigned int NUMBER_OF_REPRESENTATIVE_FIBERS;
    float SIDE_SIZE; //mm

    //Rendering
    bool SHOW_FIBER_SAMPLES;
    bool SHOW_REPRESENTATIVE_FIBERS;

    bool USE_TRILINEAR_INTERPOLATION;
    bool USE_FIBER_FREQUENCIES; //when false it will use distance scores

    float  ISOVALUE_MIN_FREQUENCY_PERCENTAGE;
    double ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE;

    float HULL_OPACITY;
    float DWI_OPACITY;

    static Configuration& getInstance()
    {
        static Configuration instance;
        return instance;
    }
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CONFIGURATION_H


