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
    OGLWidget::connect(mainWindow.showFiberSamplesCheckBox, &QCheckBox::clicked, this, &UserInterface::showFiberSamplesClicked);
    OGLWidget::connect(mainWindow.showRepresentativeFibersCheckBox, &QCheckBox::clicked, this, &UserInterface::showRepresentativeFibersClicked);
    OGLWidget::connect(mainWindow.useTrilinearInterpolationCheckBox, &QCheckBox::clicked, this, &UserInterface::useTrilinearInterpolationClicked);
    OGLWidget::connect(mainWindow.fiberFrequenciesRadioButton, &QRadioButton::clicked, this, &UserInterface::useFiberFrequenciesClicked);
    OGLWidget::connect(mainWindow.distanceScoresRadioButton, &QRadioButton::clicked, this, &UserInterface::useDistanceScoresClicked);
    OGLWidget::connect(mainWindow.fiberFrequencySlider, &QSlider::valueChanged, this, &UserInterface::fiberFrequencySliderValueChanged);
    OGLWidget::connect(mainWindow.distanceScoreSlider, &QSlider::valueChanged, this, &UserInterface::distanceScoreSliderValueChanged);
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

    mainWindow.showFiberSamplesCheckBox->setChecked(config.SHOW_FIBER_SAMPLES);
    mainWindow.showRepresentativeFibersCheckBox->setChecked(config.SHOW_REPRESENTATIVE_FIBERS);
    mainWindow.useTrilinearInterpolationCheckBox->setChecked(config.USE_TRILINEAR_INTERPOLATION);
    mainWindow.fiberFrequenciesRadioButton->setChecked(config.USE_FIBER_FREQUENCIES);
    mainWindow.distanceScoresRadioButton->setChecked(!config.USE_FIBER_FREQUENCIES);

    int percentage = (int)config.ISOVALUE_MIN_FREQUENCY_PERCENTAGE * 100;
    mainWindow.fiberFrequencySlider->setValue(100 - percentage);
    mainWindow.fiberFrequencyLabel->setText(QString::number(percentage));

    //TODO: fix distance score slider
    mainWindow.distanceScoreSlider->setValue(config.ISOVALUE_MAX_DISTANCE_SCORE);

    mainWindow.fiberFrequencyWidget->setVisible(config.USE_FIBER_FREQUENCIES);
    mainWindow.distanceScoreWidget->setVisible(!config.USE_FIBER_FREQUENCIES);
}

void UserInterface::startButtonClicked()
{
    std::cout << "Clicked!" << std::endl;
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
    Configuration::getInstance().ISOVALUE_MIN_FREQUENCY_PERCENTAGE = percentage / 100.0f;
}

void UserInterface::distanceScoreSliderValueChanged(int value)
{

}
