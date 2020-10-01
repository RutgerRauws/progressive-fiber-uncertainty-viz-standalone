//
// Created by rutger on 7/28/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_OBSERVER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_OBSERVER_H


class KeyPressObserver
{
    public:
        virtual void KeyPressed(const std::basic_string<char>& value) = 0;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_KEY_PRESS_OBSERVER_H
