//
// Created by rutger on 11/16/20.
//

#include "UserInterface.h"

UserInterface::UserInterface()
    : window(),
      mainWindow()
{
    mainWindow.setupUi(&window);
}

int UserInterface::Show()
{
    window.show();
}

OGLWidget* UserInterface::GetOpenGLWidget()
{
    return mainWindow.openGLWidget;
}

