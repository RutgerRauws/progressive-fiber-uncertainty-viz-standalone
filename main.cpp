#include "main.h"

#include <src/gui/UserInterface.h>
#include <QtOpenGL/qgl.h>
#include "src/interaction/InteractionManager.h"


int main(int argc, char* argv[])
{
    QApplication a(argc, argv);

    UserInterface ui;

    OGLWidget* widget = ui.GetOpenGLWidget();
//    widget->setFormat(glFormat);
    const QOpenGLContext* context = widget->context();

    ui.Show();


    return a.exec();
}