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

    //Setup
    void startButtonClicked();

    //General
    void showAxialPlaneClicked(bool checked);
    void showCoronalPlaneClicked(bool checked);
    void showSagittalPlaneClicked(bool checked);

    void showFiberSamplesClicked(bool checked);
    void showRepresentativeFibersClicked(bool checked);

    //Basic visitation map rendering settings
    void useTrilinearInterpolationClicked(bool checked);
    void useFiberFrequenciesClicked(bool checked);
    void useDistanceScoresClicked(bool checked);

    //Hull related
    void hullFiberFrequencySliderValueChanged(int value);
    void hullDistanceScoreSliderValueChanged(int value);
    void hullOpacitySliderValueChanged(int value);
    void hullDiffuseColorPickerClicked(bool checked);
    void hullAmbientColorPickerClicked(bool checked);
    void hullSpecularColorPickerClicked(bool checked);

    //Silhouette related
    void silhouetteFiberFrequencySliderValueChanged(int value);
    void silhouetteDistanceScoreSliderValueChanged(int value);
    void silhouetteOpacitySliderValueChanged(int value);
    void silhouetteColorPickerClicked(bool checked);

public:
    UserInterface();

    void Show();
    OGLWidget* GetOpenGLWidget();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_USER_INTERFACE_H
