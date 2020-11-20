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
        NUMBER_OF_REPRESENTATIVE_FIBERS = 15;
        SIDE_SIZE = 1.0f;

        SHOW_FIBER_SAMPLES = true;
        SHOW_REPRESENTATIVE_FIBERS = true;

        USE_TRILINEAR_INTERPOLATION = false;
        USE_FIBER_FREQUENCIES = true;

        ISOVALUE_MIN_FREQUENCY_PERCENTAGE = 0; //0%
        ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE = 1; //100%

        OPACITY = 1.0f; //fully opaque
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

    float OPACITY;

    static Configuration& getInstance()
    {
        static Configuration instance;
        return instance;
    }
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_CONFIGURATION_H


