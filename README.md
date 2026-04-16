Basic PyGame visualisation of an inverted pendulum, solved using Active Inference controls.

The command $ python3 invert.py opens up an inverted pendulum in manual mode whose torque can be controlled using the left and right arrow keys (+15Nm & -15Nm clockwise for left and right respectively).
Once the state of the pendulum is set, it can be switched to ActiveInference (AI) mode using the tab key, post which it automatically balances at the 0 degree mark. 

In the AI mode, the state of the pendulum can be changed manually as well, where we use a control law of 85% human input + 15% AI damping to smoothen the results.

Graphs have been added along the screen to aid visualisation of the system states.
