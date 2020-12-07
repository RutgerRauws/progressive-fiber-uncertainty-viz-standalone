//
// Created by rutger on 11/16/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_USER_INTERFACE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_USER_INTERFACE_H

#include <QApplication>
#include "ui/MainWindow.h"

class UserInterface : public QObject
{
private:
    QMainWindow window;
    Ui::MainWindow mainWindow;

    void loadConfiguration();

    void startButtonClicked();

    void showDWISlicesClicked(bool checked);
    void showFiberSamplesClicked(bool checked);
    void showRepresentativeFibersClicked(bool checked);

    void useTrilinearInterpolationClicked(bool checked);
    void useFiberFrequenciesClicked(bool checked);
    void useDistanceScoresClicked(bool checked);
    void fiberFrequencySliderValueChanged(int value);
    void distanceScoreSliderValueChanged(int value);
    void hullOpacitySliderValueChanged(int value);
    void diffuseColorSelectButtonClicked(bool checked);
    void ambientColorSelectButtonClicked(bool checked);
    void specularColorSelectButtonClicked(bool checked);

public:
    UserInterface();

    void Show();
    OGLWidget* GetOpenGLWidget();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_USER_INTERFACE_H
