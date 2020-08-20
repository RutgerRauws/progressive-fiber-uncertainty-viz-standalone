//
// Created by rutger on 7/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <limits.h>
#include <map>
#include "KeyPressObserver.h"

class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
private:
    std::map<std::basic_string<char>, KeyPressObserver*> observers;

public:
    static KeyPressInteractorStyle* New();
    vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera);

    void OnChar() override
    {
        // By overriding this function and not calling the base method we do not forward the events,
        // and no VTK-specific default keypresses are handled
    }

    void OnKeyPress() override
    {
        // Get the keypress
        vtkRenderWindowInteractor *rwi = this->Interactor;
        std::string key = rwi->GetKeySym();

        if(key == "q")
        {
            std::cout << "The exit key was pressed." << std::endl;

            rwi->GetRenderWindow()->Finalize();
            rwi->TerminateApp();
            return;
        }

        try
        {
            observers.at(key)->KeyPressed(key);
        }
        catch (const std::out_of_range& exception)
        {
            //Key is not handled
        }

        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }

    void AddObserver(const std::basic_string<char>& key, KeyPressObserver* observer)
    {
        observers[key] = observer;
    }
};

vtkStandardNewMacro(KeyPressInteractorStyle);

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H
