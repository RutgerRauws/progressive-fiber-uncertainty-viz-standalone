/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.1
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
    QWidget *tab_3;
    QVBoxLayout *verticalLayout_7;
    QGroupBox *groupBox_5;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_2;
    QSpacerItem *horizontalSpacer;
    QDoubleSpinBox *sideSizeDoubleSpinBox;
    QLabel *label_3;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QSpacerItem *horizontalSpacer_2;
    QSpinBox *numberOfRepresentativeFibersSpinBox;
    QSpacerItem *verticalSpacer_3;
    QPushButton *startButton;
    QWidget *tab_4;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *showFiberSamplesCheckBox;
    QCheckBox *showRepresentativeFibersCheckBox;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_3;
    QCheckBox *useTrilinearInterpolationCheckBox;
    QLabel *label_4;
    QHBoxLayout *horizontalLayout_6;
    QRadioButton *fiberFrequenciesRadioButton;
    QRadioButton *distanceScoresRadioButton;
    QLabel *label_5;
    QSlider *isovalueSlider;
    QLabel *label_6;
    QSlider *distanceScoreSlider;
    QSpacerItem *verticalSpacer;
    QButtonGroup *buttonGroup;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1172, 648);
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
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        verticalLayout_7 = new QVBoxLayout(tab_3);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        groupBox_5 = new QGroupBox(tab_3);
        groupBox_5->setObjectName(QString::fromUtf8("groupBox_5"));
        verticalLayout_5 = new QVBoxLayout(groupBox_5);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_2 = new QLabel(groupBox_5);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_4->addWidget(label_2);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);

        sideSizeDoubleSpinBox = new QDoubleSpinBox(groupBox_5);
        sideSizeDoubleSpinBox->setObjectName(QString::fromUtf8("sideSizeDoubleSpinBox"));
        sideSizeDoubleSpinBox->setDecimals(3);
        sideSizeDoubleSpinBox->setSingleStep(0.250000000000000);
        sideSizeDoubleSpinBox->setValue(1.000000000000000);

        horizontalLayout_4->addWidget(sideSizeDoubleSpinBox);

        label_3 = new QLabel(groupBox_5);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_4->addWidget(label_3);


        verticalLayout_5->addLayout(horizontalLayout_4);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(groupBox_5);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        numberOfRepresentativeFibersSpinBox = new QSpinBox(groupBox_5);
        numberOfRepresentativeFibersSpinBox->setObjectName(QString::fromUtf8("numberOfRepresentativeFibersSpinBox"));
        numberOfRepresentativeFibersSpinBox->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);
        numberOfRepresentativeFibersSpinBox->setValue(15);

        horizontalLayout->addWidget(numberOfRepresentativeFibersSpinBox);


        verticalLayout_5->addLayout(horizontalLayout);


        verticalLayout_7->addWidget(groupBox_5);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_7->addItem(verticalSpacer_3);

        startButton = new QPushButton(tab_3);
        startButton->setObjectName(QString::fromUtf8("startButton"));

        verticalLayout_7->addWidget(startButton);

        tabWidget->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QString::fromUtf8("tab_4"));
        verticalLayout = new QVBoxLayout(tab_4);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox = new QGroupBox(tab_4);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout_2 = new QVBoxLayout(groupBox);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        showFiberSamplesCheckBox = new QCheckBox(groupBox);
        showFiberSamplesCheckBox->setObjectName(QString::fromUtf8("showFiberSamplesCheckBox"));
        showFiberSamplesCheckBox->setChecked(false);

        verticalLayout_2->addWidget(showFiberSamplesCheckBox);

        showRepresentativeFibersCheckBox = new QCheckBox(groupBox);
        showRepresentativeFibersCheckBox->setObjectName(QString::fromUtf8("showRepresentativeFibersCheckBox"));
        showRepresentativeFibersCheckBox->setChecked(false);

        verticalLayout_2->addWidget(showRepresentativeFibersCheckBox);


        verticalLayout->addWidget(groupBox);

        groupBox_2 = new QGroupBox(tab_4);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        verticalLayout_3 = new QVBoxLayout(groupBox_2);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        useTrilinearInterpolationCheckBox = new QCheckBox(groupBox_2);
        useTrilinearInterpolationCheckBox->setObjectName(QString::fromUtf8("useTrilinearInterpolationCheckBox"));

        verticalLayout_3->addWidget(useTrilinearInterpolationCheckBox);

        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        verticalLayout_3->addWidget(label_4);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        fiberFrequenciesRadioButton = new QRadioButton(groupBox_2);
        buttonGroup = new QButtonGroup(MainWindow);
        buttonGroup->setObjectName(QString::fromUtf8("buttonGroup"));
        buttonGroup->addButton(fiberFrequenciesRadioButton);
        fiberFrequenciesRadioButton->setObjectName(QString::fromUtf8("fiberFrequenciesRadioButton"));

        horizontalLayout_6->addWidget(fiberFrequenciesRadioButton);

        distanceScoresRadioButton = new QRadioButton(groupBox_2);
        buttonGroup->addButton(distanceScoresRadioButton);
        distanceScoresRadioButton->setObjectName(QString::fromUtf8("distanceScoresRadioButton"));

        horizontalLayout_6->addWidget(distanceScoresRadioButton);


        verticalLayout_3->addLayout(horizontalLayout_6);

        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        verticalLayout_3->addWidget(label_5);

        isovalueSlider = new QSlider(groupBox_2);
        isovalueSlider->setObjectName(QString::fromUtf8("isovalueSlider"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(isovalueSlider->sizePolicy().hasHeightForWidth());
        isovalueSlider->setSizePolicy(sizePolicy);
        isovalueSlider->setMaximumSize(QSize(16777215, 16777215));
        isovalueSlider->setMaximum(100);
        isovalueSlider->setOrientation(Qt::Horizontal);
        isovalueSlider->setInvertedAppearance(true);

        verticalLayout_3->addWidget(isovalueSlider);

        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        verticalLayout_3->addWidget(label_6);

        distanceScoreSlider = new QSlider(groupBox_2);
        distanceScoreSlider->setObjectName(QString::fromUtf8("distanceScoreSlider"));
        distanceScoreSlider->setOrientation(Qt::Horizontal);

        verticalLayout_3->addWidget(distanceScoreSlider);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);


        verticalLayout->addWidget(groupBox_2);

        tabWidget->addTab(tab_4, QString());

        verticalLayout_6->addWidget(tabWidget);

        dockWidget->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(Qt::LeftDockWidgetArea, dockWidget);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuHelp->menuAction());
        menuFile->addAction(actionLoad_sample_data);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "Progressive fiber uncertainty visualization", nullptr));
        actionLoad_sample_data->setText(QCoreApplication::translate("MainWindow", "Load sample data..", nullptr));
        menuFile->setTitle(QCoreApplication::translate("MainWindow", "File", nullptr));
        menuHelp->setTitle(QCoreApplication::translate("MainWindow", "Help", nullptr));
        dockWidget->setWindowTitle(QCoreApplication::translate("MainWindow", "Configuration", nullptr));
        groupBox_5->setTitle(QCoreApplication::translate("MainWindow", "Cell properties", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "Side size:", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "mm", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Number of representative fibers:", nullptr));
        startButton->setText(QCoreApplication::translate("MainWindow", "Start", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QCoreApplication::translate("MainWindow", "Setup", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "General", nullptr));
        showFiberSamplesCheckBox->setText(QCoreApplication::translate("MainWindow", "Show fiber samples", nullptr));
        showRepresentativeFibersCheckBox->setText(QCoreApplication::translate("MainWindow", "Show representative fibers", nullptr));
        groupBox_2->setTitle(QCoreApplication::translate("MainWindow", "Visitation Map", nullptr));
        useTrilinearInterpolationCheckBox->setText(QCoreApplication::translate("MainWindow", "Use trilinear interpolation", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Preferred metric:", nullptr));
        fiberFrequenciesRadioButton->setText(QCoreApplication::translate("MainWindow", "Use fiber frequencies", nullptr));
        distanceScoresRadioButton->setText(QCoreApplication::translate("MainWindow", "Use distance scores", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", "Fiber frequency threshold:", nullptr));
        label_6->setText(QCoreApplication::translate("MainWindow", "Distance score threshold:", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_4), QCoreApplication::translate("MainWindow", "Rendering", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
