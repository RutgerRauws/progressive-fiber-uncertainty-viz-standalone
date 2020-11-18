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
    void showFiberSamplesClicked(bool checked);
    void showRepresentativeFibersClicked(bool checked);
    void useTrilinearInterpolationClicked(bool checked);
    void useFiberFrequenciesClicked(bool checked);
    void useDistanceScoresClicked(bool checked);
    void fiberFrequencySliderValueChanged(int value);
    void distanceScoreSliderValueChanged(int value);

public:
    UserInterface();

    void Show();
    OGLWidget* GetOpenGLWidget();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_USER_INTERFACE_H
