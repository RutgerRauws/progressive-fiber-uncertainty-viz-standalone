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

    OGLWidget::connect(mainWindow.startButton, &QPushButton::clicked, this, &UserInterface::startButtonClicked);

    OGLWidget::connect(mainWindow.showAxialPlaneCheckBox, &QCheckBox::clicked, this, &UserInterface::showAxialPlaneClicked);
    OGLWidget::connect(mainWindow.showCoronalPlaneCheckBox, &QCheckBox::clicked, this, &UserInterface::showCoronalPlaneClicked);
    OGLWidget::connect(mainWindow.showSagittalPlaneCheckBox, &QCheckBox::clicked, this, &UserInterface::showSagittalPlaneClicked);

    OGLWidget::connect(mainWindow.showFiberSamplesCheckBox, &QCheckBox::clicked, this, &UserInterface::showFiberSamplesClicked);
    OGLWidget::connect(mainWindow.showRepresentativeFibersCheckBox, &QCheckBox::clicked, this, &UserInterface::showRepresentativeFibersClicked);

    OGLWidget::connect(mainWindow.useTrilinearInterpolationCheckBox, &QCheckBox::clicked, this, &UserInterface::useTrilinearInterpolationClicked);
    OGLWidget::connect(mainWindow.fiberFrequenciesRadioButton, &QRadioButton::clicked, this, &UserInterface::useFiberFrequenciesClicked);
    OGLWidget::connect(mainWindow.distanceScoresRadioButton, &QRadioButton::clicked, this, &UserInterface::useDistanceScoresClicked);
    OGLWidget::connect(mainWindow.fiberFrequencySlider, &QSlider::valueChanged, this, &UserInterface::fiberFrequencySliderValueChanged);
    OGLWidget::connect(mainWindow.distanceScoreSlider, &QSlider::valueChanged, this, &UserInterface::distanceScoreSliderValueChanged);

    OGLWidget::connect(mainWindow.hullOpacitySlider, &QSlider::valueChanged, this, &UserInterface::hullOpacitySliderValueChanged);
    OGLWidget::connect(mainWindow.diffuseColorSelectButton, &QPushButton::clicked, this, &UserInterface::diffuseColorSelectButtonClicked);
    OGLWidget::connect(mainWindow.ambientColorSelectButton, &QPushButton::clicked, this, &UserInterface::ambientColorSelectButtonClicked);
    OGLWidget::connect(mainWindow.specularColorSelectButton, &QPushButton::clicked, this, &UserInterface::specularColorSelectButtonClicked);
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

    mainWindow.sideSizeDoubleSpinBox->setValue(config.SIDE_SIZE);
    mainWindow.numberOfRepresentativeFibersSpinBox->setValue(config.NUMBER_OF_REPRESENTATIVE_FIBERS);

    mainWindow.showAxialPlaneCheckBox->setChecked(config.SHOW_AXIAL_PLANE);
    mainWindow.showCoronalPlaneCheckBox->setChecked(config.SHOW_CORONAL_PLANE);
    mainWindow.showSagittalPlaneCheckBox->setChecked(config.SHOW_SAGITTAL_PLANE);

    mainWindow.showFiberSamplesCheckBox->setChecked(config.SHOW_FIBER_SAMPLES);
    mainWindow.showRepresentativeFibersCheckBox->setChecked(config.SHOW_REPRESENTATIVE_FIBERS);

    mainWindow.useTrilinearInterpolationCheckBox->setChecked(config.USE_TRILINEAR_INTERPOLATION);
    mainWindow.fiberFrequenciesRadioButton->setChecked(config.USE_FIBER_FREQUENCIES);
    mainWindow.distanceScoresRadioButton->setChecked(!config.USE_FIBER_FREQUENCIES);

    int percentage = (int)(config.ISOVALUE_MIN_FREQUENCY_PERCENTAGE * 100.0f);
    mainWindow.fiberFrequencySlider->setValue(100 - percentage);
    mainWindow.fiberFrequencyLabel->setText(QString::number(percentage));

    percentage = (int)(config.ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE * 100.0f);
    mainWindow.distanceScoreSlider->setValue(percentage);
    mainWindow.distanceScoreLabel->setText(QString::number(percentage));

    mainWindow.fiberFrequencyWidget->setVisible(config.USE_FIBER_FREQUENCIES);
    mainWindow.distanceScoreWidget->setVisible(!config.USE_FIBER_FREQUENCIES);

    percentage = (int)(config.HULL_OPACITY * 100.0f);
    mainWindow.hullOpacitySlider->setValue(percentage);

    mainWindow.diffuseColorSelectButton->SetColor(Configuration::getInstance().HULL_COLOR_DIFFUSE);
    mainWindow.ambientColorSelectButton->SetColor(Configuration::getInstance().HULL_COLOR_AMBIENT);
    mainWindow.specularColorSelectButton->SetColor(Configuration::getInstance().HULL_COLOR_SPECULAR);
}

void UserInterface::startButtonClicked()
{
    std::cout << "Clicked!" << std::endl;
}

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

void UserInterface::useTrilinearInterpolationClicked(bool checked)
{
    Configuration::getInstance().USE_TRILINEAR_INTERPOLATION = checked;
}

void UserInterface::useFiberFrequenciesClicked(bool checked)
{
    Configuration::getInstance().USE_FIBER_FREQUENCIES = checked;

    mainWindow.distanceScoreWidget->setVisible(false);
    mainWindow.fiberFrequencyWidget->setVisible(true);
}

void UserInterface::useDistanceScoresClicked(bool checked)
{
    Configuration::getInstance().USE_FIBER_FREQUENCIES = !checked;

    mainWindow.fiberFrequencyWidget->setVisible(false);
    mainWindow.distanceScoreWidget->setVisible(true);
}

void UserInterface::fiberFrequencySliderValueChanged(int value)
{
    int percentage = 100 - value;

    mainWindow.fiberFrequencyLabel->setText(QString::number(percentage));
    Configuration::getInstance().ISOVALUE_MIN_FREQUENCY_PERCENTAGE = (float)percentage / 100.0f;
}

void UserInterface::distanceScoreSliderValueChanged(int value)
{
    mainWindow.distanceScoreLabel->setText(QString::number(value));
    Configuration::getInstance().ISOVALUE_MAX_DISTANCE_SCORE_PERCENTAGE = (float)value / 100.0f;
}

void UserInterface::hullOpacitySliderValueChanged(int value)
{
    Configuration::getInstance().HULL_OPACITY = (float)value / 100.0f;
}

void UserInterface::diffuseColorSelectButtonClicked(bool checked)
{
    Configuration::getInstance().HULL_COLOR_DIFFUSE = mainWindow.diffuseColorSelectButton->GetColor();
}

void UserInterface::ambientColorSelectButtonClicked(bool checked)
{
    Configuration::getInstance().HULL_COLOR_AMBIENT = mainWindow.ambientColorSelectButton->GetColor();
}

void UserInterface::specularColorSelectButtonClicked(bool checked)
{
    Configuration::getInstance().HULL_COLOR_SPECULAR = mainWindow.specularColorSelectButton->GetColor();
}
