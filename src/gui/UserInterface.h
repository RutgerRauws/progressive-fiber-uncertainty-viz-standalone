//
// Created by rutger on 11/16/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_USER_INTERFACE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_USER_INTERFACE_H

#include <QApplication>
#include "ui/MainWindow.h"

class UserInterface
{
private:
    QMainWindow window;
    Ui::MainWindow mainWindow;


public:
    UserInterface();

    int Show();
    OGLWidget* GetOpenGLWidget();
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_USER_INTERFACE_H
