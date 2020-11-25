//
// Created by rutger on 11/25/20.
//

#include "SelectColorButton.h"
#include <QColorDialog>

SelectColorButton::SelectColorButton(QWidget* parent)
{
    connect( this, SIGNAL(clicked()), this, SLOT(ChangeColor()) );
}

void SelectColorButton::UpdateColor()
{
    setStyleSheet("background-color: " + color.name());
}

void SelectColorButton::ChangeColor()
{
    QColor newColor = QColorDialog::getColor(color,parentWidget());

    if(newColor != color)
    {
        SetColor(newColor);
    }
}

void SelectColorButton::SetColor(const QColor& color )
{
    this->color = color;
    UpdateColor();
}

const QColor& SelectColorButton::GetColor()
{
    return color;
}