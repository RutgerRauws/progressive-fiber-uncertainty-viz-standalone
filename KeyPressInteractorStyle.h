//
// Created by rutger on 7/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <iostream>
#include <limits.h>
#include <stdexcept>
#include <map>
#include "KeyPressObserver.h"

class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
private:
    std::map<std::basic_string<char>, KeyPressObserver*> observers;

public:
    static KeyPressInteractorStyle* New();
    vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera);

    void OnKeyPress() override
    {
        // Get the keypress
        vtkRenderWindowInteractor *rwi = this->Interactor;
        std::string key = rwi->GetKeySym();

        //TODO: figure out why lines below cannot be used for 3D Slicer build to succeed.
        #ifdef STANDALONE
        try
        {
            observers.at(key)->KeyPressed(key);
        }
        catch (const std::out_of_range&)
        {
            //Key is not handled
        }
        #endif

        // Forward events
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }

    void AddObserver(const std::basic_string<char>& key, KeyPressObserver* observer)
    {
        observers[key] = observer;
    }

    void KeyPressed(std::basic_string<char> key)
    {
        std::cout << "The exit key was pressed." << std::endl;
        vtkRenderWindowInteractor *rwi = this->Interactor;

        rwi->GetRenderWindow()->Finalize();
        rwi->TerminateApp();
    }
    
};

vtkStandardNewMacro(KeyPressInteractorStyle);

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H
