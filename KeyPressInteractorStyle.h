//
// Created by rutger on 7/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include "IsovalueObserver.h"

class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
private:
    int isovalue = 1;
    std::vector<std::reference_wrapper<IsovalueObserver>> isovalueObservers;

    void notifyObservers(int value)
    {
        for(IsovalueObserver& observer : isovalueObservers)
        {
            observer.NewIsovalue(value);
        }
    }

public:
    static KeyPressInteractorStyle* New();
    vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera);
    
    void OnKeyPress() override
    {
        // Get the keypress
        vtkRenderWindowInteractor *rwi = this->Interactor;
        std::string key = rwi->GetKeySym();
        
        // Handle a "normal" key
        if(key == "q")
        {
            std::cout << "The exit key was pressed." << std::endl;

            rwi->GetRenderWindow()->Finalize();
            rwi->TerminateApp();
        }

        if(key == "u")
        {
            notifyObservers(++isovalue);
        }

        if(key == "j")
        {
            notifyObservers(--isovalue);
        }
        
        // Forward events
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }

    void AddObserver(IsovalueObserver& observer)
    {
        isovalueObservers.emplace_back(observer);
    }
    
};
vtkStandardNewMacro(KeyPressInteractorStyle);

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H
