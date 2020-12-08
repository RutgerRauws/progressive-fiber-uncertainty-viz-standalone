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
#include <QtWidgets/QFrame>
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
    QGroupBox *dwiSlicesGroupBox;
    QHBoxLayout *horizontalLayout_5;
    QCheckBox *showAxialPlaneCheckBox;
    QCheckBox *showCoronalPlaneCheckBox;
    QCheckBox *showSagittalPlaneCheckBox;
    QGroupBox *fibersGroupBox;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *showFiberSamplesCheckBox;
    QCheckBox *showRepresentativeFibersCheckBox;
    QGroupBox *visitationMapGroupBox;
    QVBoxLayout *verticalLayout_3;
    QCheckBox *useTrilinearInterpolationCheckBox;
    QLabel *label_4;
    QHBoxLayout *preferredMetricLayout;
    QRadioButton *fiberFrequenciesRadioButton;
    QRadioButton *distanceScoresRadioButton;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_4;
    QWidget *hullFiberFrequencyWidget;
    QVBoxLayout *fiberFrequencyLayout;
    QLabel *label_5;
    QHBoxLayout *hullFiberFrequencySliderLayout;
    QSlider *hullFiberFrequencySlider;
    QLabel *label_8;
    QLabel *hullFiberFrequencyLabel;
    QLabel *label_7;
    QWidget *hullDistanceScoreWidget;
    QVBoxLayout *distanceScoreLayout;
    QLabel *label_6;
    QHBoxLayout *hullDistanceScoreSliderLayout;
    QSlider *hullDistanceScoreSlider;
    QLabel *label_11;
    QLabel *hullDistanceScoreLabel;
    QLabel *label_9;
    QFrame *line;
    QLabel *label_10;
    QSlider *hullOpacitySlider;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_14;
    QSpacerItem *horizontalSpacer_4;
    SelectColorButton *hullDiffuseColorPicker;
    QHBoxLayout *horizontalLayout;
    QLabel *label_13;
    QSpacerItem *horizontalSpacer_3;
    SelectColorButton *hullAmbientColorPicker;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_15;
    QSpacerItem *horizontalSpacer_5;
    SelectColorButton *hullSpecularColorPicker;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_8;
    QWidget *silhouetteFiberFrequencyWidget;
    QVBoxLayout *verticalLayout_9;
    QLabel *label_12;
    QHBoxLayout *horizontalLayout_6;
    QSlider *silhouetteFiberFrequencySlider;
    QLabel *label_16;
    QLabel *silhouetteFiberFrequencyLabel;
    QLabel *label_18;
    QWidget *silhouetteDistanceScoreWidget;
    QVBoxLayout *verticalLayout_11;
    QLabel *label_19;
    QHBoxLayout *horizontalLayout_7;
    QSlider *silhouetteDistanceScoreSlider;
    QLabel *label_20;
    QLabel *silhouetteDistanceScoreLabel;
    QLabel *label_22;
    QFrame *line_2;
    QLabel *label_23;
    QSlider *silhouetteOpacitySlider;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_24;
    QSpacerItem *horizontalSpacer_6;
    SelectColorButton *silhouetteColorPicker;
    QSpacerItem *verticalSpacer;
    QButtonGroup *buttonGroup;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1172, 1044);
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
        dwiSlicesGroupBox = new QGroupBox(renderingTab);
        dwiSlicesGroupBox->setObjectName(QString::fromUtf8("dwiSlicesGroupBox"));
        horizontalLayout_5 = new QHBoxLayout(dwiSlicesGroupBox);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        showAxialPlaneCheckBox = new QCheckBox(dwiSlicesGroupBox);
        showAxialPlaneCheckBox->setObjectName(QString::fromUtf8("showAxialPlaneCheckBox"));

        horizontalLayout_5->addWidget(showAxialPlaneCheckBox);

        showCoronalPlaneCheckBox = new QCheckBox(dwiSlicesGroupBox);
        showCoronalPlaneCheckBox->setObjectName(QString::fromUtf8("showCoronalPlaneCheckBox"));

        horizontalLayout_5->addWidget(showCoronalPlaneCheckBox);

        showSagittalPlaneCheckBox = new QCheckBox(dwiSlicesGroupBox);
        showSagittalPlaneCheckBox->setObjectName(QString::fromUtf8("showSagittalPlaneCheckBox"));

        horizontalLayout_5->addWidget(showSagittalPlaneCheckBox);


        verticalLayout->addWidget(dwiSlicesGroupBox);

        fibersGroupBox = new QGroupBox(renderingTab);
        fibersGroupBox->setObjectName(QString::fromUtf8("fibersGroupBox"));
        verticalLayout_2 = new QVBoxLayout(fibersGroupBox);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        showFiberSamplesCheckBox = new QCheckBox(fibersGroupBox);
        showFiberSamplesCheckBox->setObjectName(QString::fromUtf8("showFiberSamplesCheckBox"));
        showFiberSamplesCheckBox->setChecked(false);

        verticalLayout_2->addWidget(showFiberSamplesCheckBox);

        showRepresentativeFibersCheckBox = new QCheckBox(fibersGroupBox);
        showRepresentativeFibersCheckBox->setObjectName(QString::fromUtf8("showRepresentativeFibersCheckBox"));
        showRepresentativeFibersCheckBox->setChecked(false);

        verticalLayout_2->addWidget(showRepresentativeFibersCheckBox);


        verticalLayout->addWidget(fibersGroupBox);

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

        groupBox = new QGroupBox(visitationMapGroupBox);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout_4 = new QVBoxLayout(groupBox);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        hullFiberFrequencyWidget = new QWidget(groupBox);
        hullFiberFrequencyWidget->setObjectName(QString::fromUtf8("hullFiberFrequencyWidget"));
        fiberFrequencyLayout = new QVBoxLayout(hullFiberFrequencyWidget);
        fiberFrequencyLayout->setObjectName(QString::fromUtf8("fiberFrequencyLayout"));
        label_5 = new QLabel(hullFiberFrequencyWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        fiberFrequencyLayout->addWidget(label_5);

        hullFiberFrequencySliderLayout = new QHBoxLayout();
        hullFiberFrequencySliderLayout->setObjectName(QString::fromUtf8("hullFiberFrequencySliderLayout"));
        hullFiberFrequencySlider = new QSlider(hullFiberFrequencyWidget);
        hullFiberFrequencySlider->setObjectName(QString::fromUtf8("hullFiberFrequencySlider"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(hullFiberFrequencySlider->sizePolicy().hasHeightForWidth());
        hullFiberFrequencySlider->setSizePolicy(sizePolicy);
        hullFiberFrequencySlider->setMaximumSize(QSize(16777215, 16777215));
        hullFiberFrequencySlider->setMinimum(0);
        hullFiberFrequencySlider->setMaximum(100);
        hullFiberFrequencySlider->setValue(0);
        hullFiberFrequencySlider->setOrientation(Qt::Horizontal);
        hullFiberFrequencySlider->setInvertedAppearance(true);
        hullFiberFrequencySlider->setInvertedControls(false);

        hullFiberFrequencySliderLayout->addWidget(hullFiberFrequencySlider);

        label_8 = new QLabel(hullFiberFrequencyWidget);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        QSizePolicy sizePolicy1(QSizePolicy::Maximum, QSizePolicy::Maximum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label_8->sizePolicy().hasHeightForWidth());
        label_8->setSizePolicy(sizePolicy1);

        hullFiberFrequencySliderLayout->addWidget(label_8);

        hullFiberFrequencyLabel = new QLabel(hullFiberFrequencyWidget);
        hullFiberFrequencyLabel->setObjectName(QString::fromUtf8("hullFiberFrequencyLabel"));
        sizePolicy1.setHeightForWidth(hullFiberFrequencyLabel->sizePolicy().hasHeightForWidth());
        hullFiberFrequencyLabel->setSizePolicy(sizePolicy1);

        hullFiberFrequencySliderLayout->addWidget(hullFiberFrequencyLabel);

        label_7 = new QLabel(hullFiberFrequencyWidget);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        sizePolicy1.setHeightForWidth(label_7->sizePolicy().hasHeightForWidth());
        label_7->setSizePolicy(sizePolicy1);

        hullFiberFrequencySliderLayout->addWidget(label_7);


        fiberFrequencyLayout->addLayout(hullFiberFrequencySliderLayout);


        verticalLayout_4->addWidget(hullFiberFrequencyWidget);

        hullDistanceScoreWidget = new QWidget(groupBox);
        hullDistanceScoreWidget->setObjectName(QString::fromUtf8("hullDistanceScoreWidget"));
        distanceScoreLayout = new QVBoxLayout(hullDistanceScoreWidget);
        distanceScoreLayout->setObjectName(QString::fromUtf8("distanceScoreLayout"));
        label_6 = new QLabel(hullDistanceScoreWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        distanceScoreLayout->addWidget(label_6);

        hullDistanceScoreSliderLayout = new QHBoxLayout();
        hullDistanceScoreSliderLayout->setObjectName(QString::fromUtf8("hullDistanceScoreSliderLayout"));
        hullDistanceScoreSlider = new QSlider(hullDistanceScoreWidget);
        hullDistanceScoreSlider->setObjectName(QString::fromUtf8("hullDistanceScoreSlider"));
        hullDistanceScoreSlider->setMaximum(100);
        hullDistanceScoreSlider->setOrientation(Qt::Horizontal);

        hullDistanceScoreSliderLayout->addWidget(hullDistanceScoreSlider);

        label_11 = new QLabel(hullDistanceScoreWidget);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        sizePolicy1.setHeightForWidth(label_11->sizePolicy().hasHeightForWidth());
        label_11->setSizePolicy(sizePolicy1);

        hullDistanceScoreSliderLayout->addWidget(label_11);

        hullDistanceScoreLabel = new QLabel(hullDistanceScoreWidget);
        hullDistanceScoreLabel->setObjectName(QString::fromUtf8("hullDistanceScoreLabel"));
        sizePolicy1.setHeightForWidth(hullDistanceScoreLabel->sizePolicy().hasHeightForWidth());
        hullDistanceScoreLabel->setSizePolicy(sizePolicy1);

        hullDistanceScoreSliderLayout->addWidget(hullDistanceScoreLabel);

        label_9 = new QLabel(hullDistanceScoreWidget);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        sizePolicy1.setHeightForWidth(label_9->sizePolicy().hasHeightForWidth());
        label_9->setSizePolicy(sizePolicy1);

        hullDistanceScoreSliderLayout->addWidget(label_9);


        distanceScoreLayout->addLayout(hullDistanceScoreSliderLayout);


        verticalLayout_4->addWidget(hullDistanceScoreWidget);

        line = new QFrame(groupBox);
        line->setObjectName(QString::fromUtf8("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        verticalLayout_4->addWidget(line);

        label_10 = new QLabel(groupBox);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        verticalLayout_4->addWidget(label_10);

        hullOpacitySlider = new QSlider(groupBox);
        hullOpacitySlider->setObjectName(QString::fromUtf8("hullOpacitySlider"));
        hullOpacitySlider->setMaximum(100);
        hullOpacitySlider->setOrientation(Qt::Horizontal);

        verticalLayout_4->addWidget(hullOpacitySlider);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_14 = new QLabel(groupBox);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        horizontalLayout_3->addWidget(label_14);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_4);

        hullDiffuseColorPicker = new SelectColorButton(groupBox);
        hullDiffuseColorPicker->setObjectName(QString::fromUtf8("hullDiffuseColorPicker"));
        hullDiffuseColorPicker->setFocusPolicy(Qt::NoFocus);

        horizontalLayout_3->addWidget(hullDiffuseColorPicker);


        verticalLayout_4->addLayout(horizontalLayout_3);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_13 = new QLabel(groupBox);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout->addWidget(label_13);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);

        hullAmbientColorPicker = new SelectColorButton(groupBox);
        hullAmbientColorPicker->setObjectName(QString::fromUtf8("hullAmbientColorPicker"));
        hullAmbientColorPicker->setFocusPolicy(Qt::NoFocus);
        hullAmbientColorPicker->setCheckable(false);
        hullAmbientColorPicker->setChecked(false);
        hullAmbientColorPicker->setFlat(false);

        horizontalLayout->addWidget(hullAmbientColorPicker);


        verticalLayout_4->addLayout(horizontalLayout);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_15 = new QLabel(groupBox);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        horizontalLayout_4->addWidget(label_15);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_5);

        hullSpecularColorPicker = new SelectColorButton(groupBox);
        hullSpecularColorPicker->setObjectName(QString::fromUtf8("hullSpecularColorPicker"));
        hullSpecularColorPicker->setFocusPolicy(Qt::NoFocus);

        horizontalLayout_4->addWidget(hullSpecularColorPicker);


        verticalLayout_4->addLayout(horizontalLayout_4);


        verticalLayout_3->addWidget(groupBox);

        groupBox_2 = new QGroupBox(visitationMapGroupBox);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        verticalLayout_8 = new QVBoxLayout(groupBox_2);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        silhouetteFiberFrequencyWidget = new QWidget(groupBox_2);
        silhouetteFiberFrequencyWidget->setObjectName(QString::fromUtf8("silhouetteFiberFrequencyWidget"));
        verticalLayout_9 = new QVBoxLayout(silhouetteFiberFrequencyWidget);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        label_12 = new QLabel(silhouetteFiberFrequencyWidget);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        verticalLayout_9->addWidget(label_12);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        silhouetteFiberFrequencySlider = new QSlider(silhouetteFiberFrequencyWidget);
        silhouetteFiberFrequencySlider->setObjectName(QString::fromUtf8("silhouetteFiberFrequencySlider"));
        silhouetteFiberFrequencySlider->setMaximum(100);
        silhouetteFiberFrequencySlider->setOrientation(Qt::Horizontal);
        silhouetteFiberFrequencySlider->setInvertedAppearance(true);
        silhouetteFiberFrequencySlider->setInvertedControls(false);

        horizontalLayout_6->addWidget(silhouetteFiberFrequencySlider);

        label_16 = new QLabel(silhouetteFiberFrequencyWidget);
        label_16->setObjectName(QString::fromUtf8("label_16"));

        horizontalLayout_6->addWidget(label_16);

        silhouetteFiberFrequencyLabel = new QLabel(silhouetteFiberFrequencyWidget);
        silhouetteFiberFrequencyLabel->setObjectName(QString::fromUtf8("silhouetteFiberFrequencyLabel"));

        horizontalLayout_6->addWidget(silhouetteFiberFrequencyLabel);

        label_18 = new QLabel(silhouetteFiberFrequencyWidget);
        label_18->setObjectName(QString::fromUtf8("label_18"));

        horizontalLayout_6->addWidget(label_18);


        verticalLayout_9->addLayout(horizontalLayout_6);


        verticalLayout_8->addWidget(silhouetteFiberFrequencyWidget);

        silhouetteDistanceScoreWidget = new QWidget(groupBox_2);
        silhouetteDistanceScoreWidget->setObjectName(QString::fromUtf8("silhouetteDistanceScoreWidget"));
        verticalLayout_11 = new QVBoxLayout(silhouetteDistanceScoreWidget);
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        label_19 = new QLabel(silhouetteDistanceScoreWidget);
        label_19->setObjectName(QString::fromUtf8("label_19"));

        verticalLayout_11->addWidget(label_19);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        silhouetteDistanceScoreSlider = new QSlider(silhouetteDistanceScoreWidget);
        silhouetteDistanceScoreSlider->setObjectName(QString::fromUtf8("silhouetteDistanceScoreSlider"));
        silhouetteDistanceScoreSlider->setMaximum(100);
        silhouetteDistanceScoreSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_7->addWidget(silhouetteDistanceScoreSlider);

        label_20 = new QLabel(silhouetteDistanceScoreWidget);
        label_20->setObjectName(QString::fromUtf8("label_20"));

        horizontalLayout_7->addWidget(label_20);

        silhouetteDistanceScoreLabel = new QLabel(silhouetteDistanceScoreWidget);
        silhouetteDistanceScoreLabel->setObjectName(QString::fromUtf8("silhouetteDistanceScoreLabel"));

        horizontalLayout_7->addWidget(silhouetteDistanceScoreLabel);

        label_22 = new QLabel(silhouetteDistanceScoreWidget);
        label_22->setObjectName(QString::fromUtf8("label_22"));

        horizontalLayout_7->addWidget(label_22);


        verticalLayout_11->addLayout(horizontalLayout_7);


        verticalLayout_8->addWidget(silhouetteDistanceScoreWidget);

        line_2 = new QFrame(groupBox_2);
        line_2->setObjectName(QString::fromUtf8("line_2"));
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);

        verticalLayout_8->addWidget(line_2);

        label_23 = new QLabel(groupBox_2);
        label_23->setObjectName(QString::fromUtf8("label_23"));

        verticalLayout_8->addWidget(label_23);

        silhouetteOpacitySlider = new QSlider(groupBox_2);
        silhouetteOpacitySlider->setObjectName(QString::fromUtf8("silhouetteOpacitySlider"));
        silhouetteOpacitySlider->setOrientation(Qt::Horizontal);

        verticalLayout_8->addWidget(silhouetteOpacitySlider);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_24 = new QLabel(groupBox_2);
        label_24->setObjectName(QString::fromUtf8("label_24"));

        horizontalLayout_8->addWidget(label_24);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(horizontalSpacer_6);

        silhouetteColorPicker = new SelectColorButton(groupBox_2);
        silhouetteColorPicker->setObjectName(QString::fromUtf8("silhouetteColorPicker"));
        silhouetteColorPicker->setFocusPolicy(Qt::NoFocus);

        horizontalLayout_8->addWidget(silhouetteColorPicker);


        verticalLayout_8->addLayout(horizontalLayout_8);


        verticalLayout_3->addWidget(groupBox_2);


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
        hullAmbientColorPicker->setDefault(true);


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
        dwiSlicesGroupBox->setTitle(QCoreApplication::translate("MainWindow", "Show planes", nullptr));
        showAxialPlaneCheckBox->setText(QCoreApplication::translate("MainWindow", "Axial", nullptr));
        showCoronalPlaneCheckBox->setText(QCoreApplication::translate("MainWindow", "Coronal", nullptr));
        showSagittalPlaneCheckBox->setText(QCoreApplication::translate("MainWindow", "Sagittal", nullptr));
        fibersGroupBox->setTitle(QCoreApplication::translate("MainWindow", "Fibers", nullptr));
        showFiberSamplesCheckBox->setText(QCoreApplication::translate("MainWindow", "Show fiber samples", nullptr));
        showRepresentativeFibersCheckBox->setText(QCoreApplication::translate("MainWindow", "Show representative fibers", nullptr));
        visitationMapGroupBox->setTitle(QCoreApplication::translate("MainWindow", "Visitation Map", nullptr));
        useTrilinearInterpolationCheckBox->setText(QCoreApplication::translate("MainWindow", "Use trilinear interpolation", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Preferred metric:", nullptr));
        fiberFrequenciesRadioButton->setText(QCoreApplication::translate("MainWindow", "Use fiber frequencies", nullptr));
        distanceScoresRadioButton->setText(QCoreApplication::translate("MainWindow", "Use distance scores", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "Hull", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", "Fiber frequency threshold:", nullptr));
        label_8->setText(QCoreApplication::translate("MainWindow", "<html><head/><body><p>&ge;</p></body></html>", nullptr));
        hullFiberFrequencyLabel->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label_7->setText(QCoreApplication::translate("MainWindow", "%", nullptr));
        label_6->setText(QCoreApplication::translate("MainWindow", "Distance score threshold:", nullptr));
        label_11->setText(QCoreApplication::translate("MainWindow", "<html><head/><body><p>&le;</p></body></html>", nullptr));
        hullDistanceScoreLabel->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label_9->setText(QCoreApplication::translate("MainWindow", "%", nullptr));
        label_10->setText(QCoreApplication::translate("MainWindow", "Opacity:", nullptr));
        label_14->setText(QCoreApplication::translate("MainWindow", "Diffuse color:", nullptr));
        hullDiffuseColorPicker->setText(QString());
        label_13->setText(QCoreApplication::translate("MainWindow", "Ambient color:", nullptr));
        hullAmbientColorPicker->setText(QString());
        label_15->setText(QCoreApplication::translate("MainWindow", "Specular color:", nullptr));
        hullSpecularColorPicker->setText(QString());
        groupBox_2->setTitle(QCoreApplication::translate("MainWindow", "Silhouette", nullptr));
        label_12->setText(QCoreApplication::translate("MainWindow", "Fiber frequency threshold:", nullptr));
        label_16->setText(QCoreApplication::translate("MainWindow", "<html><head/><body><p>&ge;</p></body></html>", nullptr));
        silhouetteFiberFrequencyLabel->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label_18->setText(QCoreApplication::translate("MainWindow", "%", nullptr));
        label_19->setText(QCoreApplication::translate("MainWindow", "Distance score threshold:", nullptr));
        label_20->setText(QCoreApplication::translate("MainWindow", "<html><head/><body><p>&le;</p></body></html>", nullptr));
        silhouetteDistanceScoreLabel->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label_22->setText(QCoreApplication::translate("MainWindow", "%", nullptr));
        label_23->setText(QCoreApplication::translate("MainWindow", "Opacity:", nullptr));
        label_24->setText(QCoreApplication::translate("MainWindow", "Color:", nullptr));
        silhouetteColorPicker->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(renderingTab), QCoreApplication::translate("MainWindow", "Rendering", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
