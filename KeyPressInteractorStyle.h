//
// Created by rutger on 7/1/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>

class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
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
            
            rwi->TerminateApp();
        }
        
        // Forward events
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }
    
};
vtkStandardNewMacro(KeyPressInteractorStyle);

#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_INTERACTOR_STYLE_H
