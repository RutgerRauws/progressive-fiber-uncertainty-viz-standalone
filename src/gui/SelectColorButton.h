//
// Created by rutger on 11/25/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SELECT_COLOR_BUTTON_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SELECT_COLOR_BUTTON_H

#include <QPushButton>
#include <QColor>

//From https://stackoverflow.com/questions/18257281/qt-color-picker-widget
class SelectColorButton : public QPushButton
{
    Q_OBJECT
public:
    explicit SelectColorButton(QWidget* parent);

    void SetColor(const QColor& color);
    const QColor& GetColor();

public slots:
    void UpdateColor();
    void ChangeColor();

private:
    QColor color;
};

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_SELECT_COLOR_BUTTON_H
