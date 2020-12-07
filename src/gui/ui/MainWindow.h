/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "src/gui/OGLWidget.h"
#include "src/gui/SelectColorButton.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionLoad_sample_data;
    QWidget *centralwidget;
    QHBoxLayout *horizontalLayout_2;
    OGLWidget *openGLWidget;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menuHelp;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_6;
    QTabWidget *tabWidget;
    QWidget *setupTab;
    QVBoxLayout *verticalLayout_7;
    QGroupBox *cellPropertiesGroupBox;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *sideSizeLayout;
    QLabel *label_2;
    QSpacerItem *horizontalSpacer;
    QDoubleSpinBox *sideSizeDoubleSpinBox;
    QLabel *label_3;
    QHBoxLayout *numberOfRepresentativeFibersLayout;
    QLabel *label;
    QSpacerItem *horizontalSpacer_2;
    QSpinBox *numberOfRepresentativeFibersSpinBox;
    QSpacerItem *verticalSpacer_3;
    QPushButton *startButton;
    QWidget *renderingTab;
    QVBoxLayout *verticalLayout;
    QGroupBox *generalGroupBox;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *showDWISlicesCheckBox;
    QCheckBox *showFiberSamplesCheckBox;
    QCheckBox *showRepresentativeFibersCheckBox;
    QGroupBox *visitationMapGroupBox;
    QVBoxLayout *verticalLayout_3;
    QCheckBox *useTrilinearInterpolationCheckBox;
    QLabel *label_4;
    QHBoxLayout *preferredMetricLayout;
    QRadioButton *fiberFrequenciesRadioButton;
    QRadioButton *distanceScoresRadioButton;
    QWidget *fiberFrequencyWidget;
    QVBoxLayout *fiberFrequencyLayout;
    QLabel *label_5;
    QHBoxLayout *fiberFrequencySliderLayout;
    QSlider *fiberFrequencySlider;
    QLabel *label_8;
    QLabel *fiberFrequencyLabel;
    QLabel *label_7;
    QWidget *distanceScoreWidget;
    QVBoxLayout *distanceScoreLayout;
    QLabel *label_6;
    QHBoxLayout *distanceScoreSliderLayout;
    QSlider *distanceScoreSlider;
    QLabel *label_11;
    QLabel *distanceScoreLabel;
    QLabel *label_9;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_4;
    QLabel *label_10;
    QSlider *hullOpacitySlider;
    QHBoxLayout *horizontalLayout;
    QLabel *label_13;
    QSpacerItem *horizontalSpacer_3;
    SelectColorButton *ambientColorSelectButton;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_14;
    QSpacerItem *horizontalSpacer_4;
    SelectColorButton *diffuseColorSelectButton;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_15;
    QSpacerItem *horizontalSpacer_5;
    SelectColorButton *specularColorSelectButton;
    QSpacerItem *verticalSpacer;
    QButtonGroup *buttonGroup;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1172, 818);
        QPalette palette;
        QBrush brush(QColor(252, 252, 252, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Base, brush);
        QBrush brush1(QColor(239, 240, 241, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Window, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush1);
        MainWindow->setPalette(palette);
        MainWindow->setAutoFillBackground(false);
        actionLoad_sample_data = new QAction(MainWindow);
        actionLoad_sample_data->setObjectName(QString::fromUtf8("actionLoad_sample_data"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        horizontalLayout_2 = new QHBoxLayout(centralwidget);
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        openGLWidget = new OGLWidget(centralwidget);
        openGLWidget->setObjectName(QString::fromUtf8("openGLWidget"));
        openGLWidget->setAutoFillBackground(false);

        horizontalLayout_2->addWidget(openGLWidget);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1172, 30));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        menuHelp = new QMenu(menubar);
        menuHelp->setObjectName(QString::fromUtf8("menuHelp"));
        MainWindow->setMenuBar(menubar);
        dockWidget = new QDockWidget(MainWindow);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        dockWidget->setStyleSheet(QString::fromUtf8(""));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        verticalLayout_6 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        tabWidget = new QTabWidget(dockWidgetContents);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setTabPosition(QTabWidget::North);
        tabWidget->setTabShape(QTabWidget::Rounded);
        tabWidget->setElideMode(Qt::ElideNone);
        tabWidget->setUsesScrollButtons(false);
        tabWidget->setTabBarAutoHide(false);
        setupTab = new QWidget();
        setupTab->setObjectName(QString::fromUtf8("setupTab"));
        verticalLayout_7 = new QVBoxLayout(setupTab);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        cellPropertiesGroupBox = new QGroupBox(setupTab);
        cellPropertiesGroupBox->setObjectName(QString::fromUtf8("cellPropertiesGroupBox"));
        verticalLayout_5 = new QVBoxLayout(cellPropertiesGroupBox);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        sideSizeLayout = new QHBoxLayout();
        sideSizeLayout->setObjectName(QString::fromUtf8("sideSizeLayout"));
        label_2 = new QLabel(cellPropertiesGroupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        sideSizeLayout->addWidget(label_2);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        sideSizeLayout->addItem(horizontalSpacer);

        sideSizeDoubleSpinBox = new QDoubleSpinBox(cellPropertiesGroupBox);
        sideSizeDoubleSpinBox->setObjectName(QString::fromUtf8("sideSizeDoubleSpinBox"));
        sideSizeDoubleSpinBox->setDecimals(3);
        sideSizeDoubleSpinBox->setSingleStep(0.250000000000000);
        sideSizeDoubleSpinBox->setValue(1.000000000000000);

        sideSizeLayout->addWidget(sideSizeDoubleSpinBox);

        label_3 = new QLabel(cellPropertiesGroupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        sideSizeLayout->addWidget(label_3);


        verticalLayout_5->addLayout(sideSizeLayout);

        numberOfRepresentativeFibersLayout = new QHBoxLayout();
        numberOfRepresentativeFibersLayout->setObjectName(QString::fromUtf8("numberOfRepresentativeFibersLayout"));
        label = new QLabel(cellPropertiesGroupBox);
        label->setObjectName(QString::fromUtf8("label"));

        numberOfRepresentativeFibersLayout->addWidget(label);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        numberOfRepresentativeFibersLayout->addItem(horizontalSpacer_2);

        numberOfRepresentativeFibersSpinBox = new QSpinBox(cellPropertiesGroupBox);
        numberOfRepresentativeFibersSpinBox->setObjectName(QString::fromUtf8("numberOfRepresentativeFibersSpinBox"));
        numberOfRepresentativeFibersSpinBox->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);
        numberOfRepresentativeFibersSpinBox->setValue(15);

        numberOfRepresentativeFibersLayout->addWidget(numberOfRepresentativeFibersSpinBox);


        verticalLayout_5->addLayout(numberOfRepresentativeFibersLayout);


        verticalLayout_7->addWidget(cellPropertiesGroupBox);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_7->addItem(verticalSpacer_3);

        startButton = new QPushButton(setupTab);
        startButton->setObjectName(QString::fromUtf8("startButton"));

        verticalLayout_7->addWidget(startButton);

        tabWidget->addTab(setupTab, QString());
        renderingTab = new QWidget();
        renderingTab->setObjectName(QString::fromUtf8("renderingTab"));
        verticalLayout = new QVBoxLayout(renderingTab);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        generalGroupBox = new QGroupBox(renderingTab);
        generalGroupBox->setObjectName(QString::fromUtf8("generalGroupBox"));
        verticalLayout_2 = new QVBoxLayout(generalGroupBox);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        showDWISlicesCheckBox = new QCheckBox(generalGroupBox);
        showDWISlicesCheckBox->setObjectName(QString::fromUtf8("showDWISlicesCheckBox"));

        verticalLayout_2->addWidget(showDWISlicesCheckBox);

        showFiberSamplesCheckBox = new QCheckBox(generalGroupBox);
        showFiberSamplesCheckBox->setObjectName(QString::fromUtf8("showFiberSamplesCheckBox"));
        showFiberSamplesCheckBox->setChecked(false);

        verticalLayout_2->addWidget(showFiberSamplesCheckBox);

        showRepresentativeFibersCheckBox = new QCheckBox(generalGroupBox);
        showRepresentativeFibersCheckBox->setObjectName(QString::fromUtf8("showRepresentativeFibersCheckBox"));
        showRepresentativeFibersCheckBox->setChecked(false);

        verticalLayout_2->addWidget(showRepresentativeFibersCheckBox);


        verticalLayout->addWidget(generalGroupBox);

        visitationMapGroupBox = new QGroupBox(renderingTab);
        visitationMapGroupBox->setObjectName(QString::fromUtf8("visitationMapGroupBox"));
        verticalLayout_3 = new QVBoxLayout(visitationMapGroupBox);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        useTrilinearInterpolationCheckBox = new QCheckBox(visitationMapGroupBox);
        useTrilinearInterpolationCheckBox->setObjectName(QString::fromUtf8("useTrilinearInterpolationCheckBox"));

        verticalLayout_3->addWidget(useTrilinearInterpolationCheckBox);

        label_4 = new QLabel(visitationMapGroupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        verticalLayout_3->addWidget(label_4);

        preferredMetricLayout = new QHBoxLayout();
        preferredMetricLayout->setObjectName(QString::fromUtf8("preferredMetricLayout"));
        fiberFrequenciesRadioButton = new QRadioButton(visitationMapGroupBox);
        buttonGroup = new QButtonGroup(MainWindow);
        buttonGroup->setObjectName(QString::fromUtf8("buttonGroup"));
        buttonGroup->addButton(fiberFrequenciesRadioButton);
        fiberFrequenciesRadioButton->setObjectName(QString::fromUtf8("fiberFrequenciesRadioButton"));

        preferredMetricLayout->addWidget(fiberFrequenciesRadioButton);

        distanceScoresRadioButton = new QRadioButton(visitationMapGroupBox);
        buttonGroup->addButton(distanceScoresRadioButton);
        distanceScoresRadioButton->setObjectName(QString::fromUtf8("distanceScoresRadioButton"));

        preferredMetricLayout->addWidget(distanceScoresRadioButton);


        verticalLayout_3->addLayout(preferredMetricLayout);

        fiberFrequencyWidget = new QWidget(visitationMapGroupBox);
        fiberFrequencyWidget->setObjectName(QString::fromUtf8("fiberFrequencyWidget"));
        fiberFrequencyLayout = new QVBoxLayout(fiberFrequencyWidget);
        fiberFrequencyLayout->setObjectName(QString::fromUtf8("fiberFrequencyLayout"));
        label_5 = new QLabel(fiberFrequencyWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        fiberFrequencyLayout->addWidget(label_5);

        fiberFrequencySliderLayout = new QHBoxLayout();
        fiberFrequencySliderLayout->setObjectName(QString::fromUtf8("fiberFrequencySliderLayout"));
        fiberFrequencySlider = new QSlider(fiberFrequencyWidget);
        fiberFrequencySlider->setObjectName(QString::fromUtf8("fiberFrequencySlider"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(fiberFrequencySlider->sizePolicy().hasHeightForWidth());
        fiberFrequencySlider->setSizePolicy(sizePolicy);
        fiberFrequencySlider->setMaximumSize(QSize(16777215, 16777215));
        fiberFrequencySlider->setMinimum(0);
        fiberFrequencySlider->setMaximum(100);
        fiberFrequencySlider->setValue(0);
        fiberFrequencySlider->setOrientation(Qt::Horizontal);
        fiberFrequencySlider->setInvertedAppearance(true);
        fiberFrequencySlider->setInvertedControls(false);

        fiberFrequencySliderLayout->addWidget(fiberFrequencySlider);

        label_8 = new QLabel(fiberFrequencyWidget);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        QSizePolicy sizePolicy1(QSizePolicy::Maximum, QSizePolicy::Maximum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label_8->sizePolicy().hasHeightForWidth());
        label_8->setSizePolicy(sizePolicy1);

        fiberFrequencySliderLayout->addWidget(label_8);

        fiberFrequencyLabel = new QLabel(fiberFrequencyWidget);
        fiberFrequencyLabel->setObjectName(QString::fromUtf8("fiberFrequencyLabel"));
        sizePolicy1.setHeightForWidth(fiberFrequencyLabel->sizePolicy().hasHeightForWidth());
        fiberFrequencyLabel->setSizePolicy(sizePolicy1);

        fiberFrequencySliderLayout->addWidget(fiberFrequencyLabel);

        label_7 = new QLabel(fiberFrequencyWidget);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        sizePolicy1.setHeightForWidth(label_7->sizePolicy().hasHeightForWidth());
        label_7->setSizePolicy(sizePolicy1);

        fiberFrequencySliderLayout->addWidget(label_7);


        fiberFrequencyLayout->addLayout(fiberFrequencySliderLayout);


        verticalLayout_3->addWidget(fiberFrequencyWidget);

        distanceScoreWidget = new QWidget(visitationMapGroupBox);
        distanceScoreWidget->setObjectName(QString::fromUtf8("distanceScoreWidget"));
        distanceScoreLayout = new QVBoxLayout(distanceScoreWidget);
        distanceScoreLayout->setObjectName(QString::fromUtf8("distanceScoreLayout"));
        label_6 = new QLabel(distanceScoreWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        distanceScoreLayout->addWidget(label_6);

        distanceScoreSliderLayout = new QHBoxLayout();
        distanceScoreSliderLayout->setObjectName(QString::fromUtf8("distanceScoreSliderLayout"));
        distanceScoreSlider = new QSlider(distanceScoreWidget);
        distanceScoreSlider->setObjectName(QString::fromUtf8("distanceScoreSlider"));
        distanceScoreSlider->setMaximum(100);
        distanceScoreSlider->setOrientation(Qt::Horizontal);

        distanceScoreSliderLayout->addWidget(distanceScoreSlider);

        label_11 = new QLabel(distanceScoreWidget);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        sizePolicy1.setHeightForWidth(label_11->sizePolicy().hasHeightForWidth());
        label_11->setSizePolicy(sizePolicy1);

        distanceScoreSliderLayout->addWidget(label_11);

        distanceScoreLabel = new QLabel(distanceScoreWidget);
        distanceScoreLabel->setObjectName(QString::fromUtf8("distanceScoreLabel"));
        sizePolicy1.setHeightForWidth(distanceScoreLabel->sizePolicy().hasHeightForWidth());
        distanceScoreLabel->setSizePolicy(sizePolicy1);

        distanceScoreSliderLayout->addWidget(distanceScoreLabel);

        label_9 = new QLabel(distanceScoreWidget);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        sizePolicy1.setHeightForWidth(label_9->sizePolicy().hasHeightForWidth());
        label_9->setSizePolicy(sizePolicy1);

        distanceScoreSliderLayout->addWidget(label_9);


        distanceScoreLayout->addLayout(distanceScoreSliderLayout);


        verticalLayout_3->addWidget(distanceScoreWidget);

        groupBox = new QGroupBox(visitationMapGroupBox);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout_4 = new QVBoxLayout(groupBox);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        label_10 = new QLabel(groupBox);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        verticalLayout_4->addWidget(label_10);

        hullOpacitySlider = new QSlider(groupBox);
        hullOpacitySlider->setObjectName(QString::fromUtf8("hullOpacitySlider"));
        hullOpacitySlider->setMaximum(100);
        hullOpacitySlider->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(hullOpacitySlider);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_13 = new QLabel(groupBox);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout->addWidget(label_13);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);

        ambientColorSelectButton = new SelectColorButton(groupBox);
        ambientColorSelectButton->setObjectName(QString::fromUtf8("ambientColorSelectButton"));
        ambientColorSelectButton->setFocusPolicy(Qt::NoFocus);
        ambientColorSelectButton->setCheckable(false);
        ambientColorSelectButton->setChecked(false);
        ambientColorSelectButton->setFlat(false);

        horizontalLayout->addWidget(ambientColorSelectButton);


        verticalLayout_4->addLayout(horizontalLayout);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_14 = new QLabel(groupBox);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        horizontalLayout_3->addWidget(label_14);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_4);

        diffuseColorSelectButton = new SelectColorButton(groupBox);
        diffuseColorSelectButton->setObjectName(QString::fromUtf8("diffuseColorSelectButton"));
        diffuseColorSelectButton->setFocusPolicy(Qt::NoFocus);

        horizontalLayout_3->addWidget(diffuseColorSelectButton);


        verticalLayout_4->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_15 = new QLabel(groupBox);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        horizontalLayout_4->addWidget(label_15);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_5);

        specularColorSelectButton = new SelectColorButton(groupBox);
        specularColorSelectButton->setObjectName(QString::fromUtf8("specularColorSelectButton"));
        specularColorSelectButton->setFocusPolicy(Qt::NoFocus);

        horizontalLayout_4->addWidget(specularColorSelectButton);


        verticalLayout_4->addLayout(horizontalLayout_4);


        verticalLayout_3->addWidget(groupBox);


        verticalLayout->addWidget(visitationMapGroupBox);

        tabWidget->addTab(renderingTab, QString());

        verticalLayout_6->addWidget(tabWidget);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer);

        dockWidget->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(Qt::LeftDockWidgetArea, dockWidget);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuHelp->menuAction());
        menuFile->addAction(actionLoad_sample_data);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(1);
        ambientColorSelectButton->setDefault(true);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "Progressive fiber uncertainty visualization", nullptr));
        actionLoad_sample_data->setText(QCoreApplication::translate("MainWindow", "Load sample data..", nullptr));
        menuFile->setTitle(QCoreApplication::translate("MainWindow", "File", nullptr));
        menuHelp->setTitle(QCoreApplication::translate("MainWindow", "Help", nullptr));
        dockWidget->setWindowTitle(QCoreApplication::translate("MainWindow", "Configuration", nullptr));
        cellPropertiesGroupBox->setTitle(QCoreApplication::translate("MainWindow", "Cell properties", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "Side size:", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "mm", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Number of representative fibers:", nullptr));
        startButton->setText(QCoreApplication::translate("MainWindow", "Start", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(setupTab), QCoreApplication::translate("MainWindow", "Setup", nullptr));
        generalGroupBox->setTitle(QCoreApplication::translate("MainWindow", "General", nullptr));
        showDWISlicesCheckBox->setText(QCoreApplication::translate("MainWindow", "Show DWI slices", nullptr));
        showFiberSamplesCheckBox->setText(QCoreApplication::translate("MainWindow", "Show fiber samples", nullptr));
        showRepresentativeFibersCheckBox->setText(QCoreApplication::translate("MainWindow", "Show representative fibers", nullptr));
        visitationMapGroupBox->setTitle(QCoreApplication::translate("MainWindow", "Visitation Map", nullptr));
        useTrilinearInterpolationCheckBox->setText(QCoreApplication::translate("MainWindow", "Use trilinear interpolation", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Preferred metric:", nullptr));
        fiberFrequenciesRadioButton->setText(QCoreApplication::translate("MainWindow", "Use fiber frequencies", nullptr));
        distanceScoresRadioButton->setText(QCoreApplication::translate("MainWindow", "Use distance scores", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", "Fiber frequency threshold:", nullptr));
        label_8->setText(QCoreApplication::translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Noto Sans'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">&ge;</p></body></html>", nullptr));
        fiberFrequencyLabel->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label_7->setText(QCoreApplication::translate("MainWindow", "%", nullptr));
        label_6->setText(QCoreApplication::translate("MainWindow", "Distance score threshold:", nullptr));
        label_11->setText(QCoreApplication::translate("MainWindow", "<html><head/><body><p>&le;</p></body></html>", nullptr));
        distanceScoreLabel->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label_9->setText(QCoreApplication::translate("MainWindow", "%", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "Appearance", nullptr));
        label_10->setText(QCoreApplication::translate("MainWindow", "Opacity:", nullptr));
        label_13->setText(QCoreApplication::translate("MainWindow", "Ambient color:", nullptr));
        ambientColorSelectButton->setText(QString());
        label_14->setText(QCoreApplication::translate("MainWindow", "Diffuse color:", nullptr));
        diffuseColorSelectButton->setText(QString());
        label_15->setText(QCoreApplication::translate("MainWindow", "Specular color:", nullptr));
        specularColorSelectButton->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(renderingTab), QCoreApplication::translate("MainWindow", "Rendering", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
