//
// Created by rutger on 11/16/20.
//

#include <Configuration.h>
#include "UserInterface.h"

UserInterface::UserInterface()
    : window(),
      mainWindow()
{
    mainWindow.setupUi(&window);

    loadConfiguration();

    //Setup
    OGLWidget::connect(mainWindow.startButton, &QPushButton::clicked, this, &UserInterface::startButtonClicked);

    //General
    OGLWidget::connect(mainWindow.showAxialPlaneCheckBox, &QCheckBox::clicked, this, &UserInterface::showAxialPlaneClicked);
    OGLWidget::connect(mainWindow.showCoronalPlaneCheckBox, &QCheckBox::clicked, this, &UserInterface::showCoronalPlaneClicked);
    OGLWidget::connect(mainWindow.showSagittalPlaneCheckBox, &QCheckBox::clicked, this, &UserInterface::showSagittalPlaneClicked);

    OGLWidget::connect(mainWindow.showFiberSamplesCheckBox, &QCheckBox::clicked, this, &UserInterface::showFiberSamplesClicked);
    OGLWidget::connect(mainWindow.showRepresentativeFibersCheckBox, &QCheckBox::clicked, this, &UserInterface::showRepresentativeFibersClicked);

    //Basic visitation map rendering settings
    OGLWidget::connect(mainWindow.useTrilinearInterpolationCheckBox, &QCheckBox::clicked, this, &UserInterface::useTrilinearInterpolationClicked);
    OGLWidget::connect(mainWindow.fiberFrequenciesRadioButton, &QRadioButton::clicked, this, &UserInterface::useFiberFrequenciesClicked);
    OGLWidget::connect(mainWindow.distanceScoresRadioButton, &QRadioButton::clicked, this, &UserInterface::useDistanceScoresClicked);

    //Hull related
    OGLWidget::connect(mainWindow.hullFiberFrequencySlider, &QSlider::valueChanged, this, &UserInterface::hullFiberFrequencySliderValueChanged);
    OGLWidget::connect(mainWindow.hullDistanceScoreSlider, &QSlider::valueChanged, this, &UserInterface::hullDistanceScoreSliderValueChanged);
    OGLWidget::connect(mainWindow.hullOpacitySlider, &QSlider::valueChanged, this, &UserInterface::hullOpacitySliderValueChanged);
    OGLWidget::connect(mainWindow.hullDiffuseColorPicker, &QPushButton::clicked, this, &UserInterface::hullDiffuseColorPickerClicked);
    OGLWidget::connect(mainWindow.hullAmbientColorPicker, &QPushButton::clicked, this, &UserInterface::hullAmbientColorPickerClicked);
    OGLWidget::connect(mainWindow.hullSpecularColorPicker, &QPushButton::clicked, this, &UserInterface::hullSpecularColorPickerClicked);

    //Silhouette related
    OGLWidget::connect(mainWindow.silhouetteFiberFrequencySlider, &QSlider::valueChanged, this, &UserInterface::silhouetteFiberFrequencySliderValueChanged);
    OGLWidget::connect(mainWindow.silhouetteDistanceScoreSlider, &QSlider::valueChanged, this, &UserInterface::silhouetteDistanceScoreSliderValueChanged);
    OGLWidget::connect(mainWindow.silhouetteOpacitySlider, &QSlider::valueChanged, this, &UserInterface::silhouetteOpacitySliderValueChanged);
    OGLWidget::connect(mainWindow.silhouetteColorPicker, &QPushButton::clicked, this, &UserInterface::silhouetteColorPickerClicked);
}


void UserInterface::Show()
{
    window.show();
}

OGLWidget* UserInterface::GetOpenGLWidget()
{
    return mainWindow.openGLWidget;
}

void UserInterface::loadConfiguration()
{
    Configuration& config = Configuration::getInstance();

    //Setup
    mainWindow.sideSizeDoubleSpinBox->setValue(config.SIDE_SIZE);
    mainWindow.numberOfRepresentativeFibersSpinBox->setValue(config.NUMBER_OF_REPRESENTATIVE_FIBERS);

    //General
    mainWindow.showAxialPlaneCheckBox->setChecked(config.SHOW_AXIAL_PLANE);
    mainWindow.showCoronalPlaneCheckBox->setChecked(config.SHOW_CORONAL_PLANE);
    mainWindow.showSagittalPlaneCheckBox->setChecked(config.SHOW_SAGITTAL_PLANE);

    mainWindow.showFiberSamplesCheckBox->setChecked(config.SHOW_FIBER_SAMPLES);
    mainWindow.showRepresentativeFibersCheckBox->setChecked(config.SHOW_REPRESENTATIVE_FIBERS);

    //Basic visitation map rendering settings
    mainWindow.useTrilinearInterpolationCheckBox->setChecked(config.USE_TRILINEAR_INTERPOLATION);
    mainWindow.fiberFrequenciesRadioButton->setChecked(config.USE_FIBER_FREQUENCIES);
    mainWindow.distanceScoresRadioButton->setChecked(!config.USE_FIBER_FREQUENCIES);

    //Hull related
    mainWindow.hullFiberFrequencyWidget->setVisible(config.USE_FIBER_FREQUENCIES);
    mainWindow.hullDistanceScoreWidget->setVisible(!config.USE_FIBER_FREQUENCIES);

    int percentage = (int)(config.HULL_ISOVALUE_MIN_FREQUENCY_PERCENTAGE * 100.0f);
    mainWindow.hullFiberFrequencySlider->setValue(100 - percentage);
    mainWindow.hullFiberFrequencyLabel->setText(QString::number(percentage));

    percentage = (int)(config.HULL_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE * 100.0f);
    mainWindow.hullDistanceScoreSlider->setValue(percentage);
    mainWindow.hullDistanceScoreLabel->setText(QString::number(percentage));

    percentage = (int)(config.HULL_OPACITY * 100.0f);
    mainWindow.hullOpacitySlider->setValue(percentage);

    mainWindow.hullDiffuseColorPicker->SetColor(config.HULL_COLOR_DIFFUSE);
    mainWindow.hullAmbientColorPicker->SetColor(config.HULL_COLOR_AMBIENT);
    mainWindow.hullSpecularColorPicker->SetColor(config.HULL_COLOR_SPECULAR);

    //Silhouette related
    mainWindow.silhouetteFiberFrequencyWidget->setVisible(config.USE_FIBER_FREQUENCIES);
    mainWindow.silhouetteDistanceScoreWidget->setVisible(!config.USE_FIBER_FREQUENCIES);

    percentage = (int)(config.SILHOUETTE_ISOVALUE_MIN_FREQUENCY_PERCENTAGE * 100.0f);
    mainWindow.silhouetteFiberFrequencySlider->setValue(100 - percentage);
    mainWindow.silhouetteFiberFrequencyLabel->setText(QString::number(percentage));

    percentage = (int)(config.SILHOUETTE_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE * 100.0f);
    mainWindow.silhouetteDistanceScoreSlider->setValue(percentage);
    mainWindow.silhouetteDistanceScoreLabel->setText(QString::number(percentage));

    percentage = (int)(config.SILHOUETTE_OPACITY * 100.0f);
    mainWindow.silhouetteOpacitySlider->setValue(percentage);
    mainWindow.silhouetteColorPicker->SetColor(config.SILHOUETTE_COLOR);
}

/*
 * Setup
 */
void UserInterface::startButtonClicked()
{
    std::cout << "Clicked!" << std::endl;
}

/*
 * General
 */
void UserInterface::showAxialPlaneClicked(bool checked)
{
    Configuration::getInstance().SHOW_AXIAL_PLANE = checked;
}

void UserInterface::showCoronalPlaneClicked(bool checked)
{
    Configuration::getInstance().SHOW_CORONAL_PLANE = checked;
}

void UserInterface::showSagittalPlaneClicked(bool checked)
{
    Configuration::getInstance().SHOW_SAGITTAL_PLANE = checked;
}

void UserInterface::showFiberSamplesClicked(bool checked)
{
    Configuration::getInstance().SHOW_FIBER_SAMPLES = checked;
}

void UserInterface::showRepresentativeFibersClicked(bool checked)
{
    Configuration::getInstance().SHOW_REPRESENTATIVE_FIBERS = checked;
}

/*
 * Basic visitation map rendering settings
 */
void UserInterface::useTrilinearInterpolationClicked(bool checked)
{
    Configuration::getInstance().USE_TRILINEAR_INTERPOLATION = checked;
}

void UserInterface::useFiberFrequenciesClicked(bool checked)
{
    Configuration::getInstance().USE_FIBER_FREQUENCIES = checked;

    mainWindow.hullDistanceScoreWidget->setVisible(false);
    mainWindow.hullFiberFrequencyWidget->setVisible(true);

    mainWindow.silhouetteDistanceScoreWidget->setVisible(false);
    mainWindow.silhouetteFiberFrequencyWidget->setVisible(true);
}

void UserInterface::useDistanceScoresClicked(bool checked)
{
    Configuration::getInstance().USE_FIBER_FREQUENCIES = !checked;

    mainWindow.hullFiberFrequencyWidget->setVisible(false);
    mainWindow.hullDistanceScoreWidget->setVisible(true);

    mainWindow.silhouetteFiberFrequencyWidget->setVisible(false);
    mainWindow.silhouetteDistanceScoreWidget->setVisible(true);
}

/*
 * Hull related
 */
void UserInterface::hullFiberFrequencySliderValueChanged(int value)
{
    if(value > mainWindow.silhouetteFiberFrequencySlider->value())
    {
        mainWindow.silhouetteFiberFrequencySlider->setValue(mainWindow.hullFiberFrequencySlider->value());
        silhouetteFiberFrequencySliderValueChanged(mainWindow.hullFiberFrequencySlider->value());
    }

    int percentage = 100 - value;

    mainWindow.hullFiberFrequencyLabel->setText(QString::number(percentage));
    Configuration::getInstance().HULL_ISOVALUE_MIN_FREQUENCY_PERCENTAGE = (float)percentage / 100.0f;
}

void UserInterface::hullDistanceScoreSliderValueChanged(int value)
{
    if(value > mainWindow.silhouetteDistanceScoreSlider->value())
    {
        mainWindow.silhouetteDistanceScoreSlider->setValue(mainWindow.hullDistanceScoreSlider->value());
        silhouetteDistanceScoreSliderValueChanged(mainWindow.hullDistanceScoreSlider->value());
    }

    mainWindow.hullDistanceScoreLabel->setText(QString::number(value));
    Configuration::getInstance().HULL_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE = (float)value / 100.0f;
}

void UserInterface::hullOpacitySliderValueChanged(int value)
{
    Configuration::getInstance().HULL_OPACITY = (float)value / 100.0f;
}

void UserInterface::hullDiffuseColorPickerClicked(bool checked)
{
    Configuration::getInstance().HULL_COLOR_DIFFUSE = mainWindow.hullDiffuseColorPicker->GetColor();
}

void UserInterface::hullAmbientColorPickerClicked(bool checked)
{
    Configuration::getInstance().HULL_COLOR_AMBIENT = mainWindow.hullAmbientColorPicker->GetColor();
}

void UserInterface::hullSpecularColorPickerClicked(bool checked)
{
    Configuration::getInstance().HULL_COLOR_SPECULAR = mainWindow.hullSpecularColorPicker->GetColor();
}

/*
 * Silhouette related
 */
void UserInterface::silhouetteFiberFrequencySliderValueChanged(int value)
{
    if(value < mainWindow.hullFiberFrequencySlider->value())
    {
        mainWindow.silhouetteFiberFrequencySlider->setValue(mainWindow.hullFiberFrequencySlider->value());
        return;
    }

    int percentage = 100 - value;

    mainWindow.silhouetteFiberFrequencyLabel->setText(QString::number(percentage));
    Configuration::getInstance().SILHOUETTE_ISOVALUE_MIN_FREQUENCY_PERCENTAGE = (float)percentage / 100.0f;
}

void UserInterface::silhouetteDistanceScoreSliderValueChanged(int value)
{
    if(value < mainWindow.hullDistanceScoreSlider->value())
    {
        mainWindow.silhouetteDistanceScoreSlider->setValue(mainWindow.hullDistanceScoreSlider->value());
        return;
    }

    mainWindow.silhouetteDistanceScoreLabel->setText(QString::number(value));
    Configuration::getInstance().SILHOUETTE_ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE = (float)value / 100.0f;
}

void UserInterface::silhouetteOpacitySliderValueChanged(int value)
{
    Configuration::getInstance().SILHOUETTE_OPACITY = (float)value / 100.0f;
}

void UserInterface::silhouetteColorPickerClicked(bool checked)
{
    Configuration::getInstance().SILHOUETTE_COLOR = mainWindow.silhouetteColorPicker->GetColor();
}