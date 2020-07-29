//
// Created by rutger on 7/28/20.
//

#ifndef PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_ISOVALUE_OBSERVER_H
#define PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_ISOVALUE_OBSERVER_H


class IsovalueObserver
{
    public:
        virtual void NewIsovalue(unsigned int value) = 0;
};


#endif //PROGRESSIVE_FIBER_UNCERTAINTY_VIZ_ISOVALUE_OBSERVER_H
