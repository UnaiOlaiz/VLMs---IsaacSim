In order to make RL applicable in the environment I had to setup the **IsaacLab** repository on top of **IsaacSim**. 

I want to work and try to apply 3 different policies:
1. [[Lift policy]]. 
2. Place policy (not available for franka).
3. Lift+place policy (not implemented for franka arm in the repository). 

Super unstable framework, the GitHub issues pages report even the tutorial do not work.

If i don't get to implement the training with inference to the pipeline, analyze and study them independently. 

Ideas:
-  Start from closer positions (franka/cube) to ease training, but does not work. 
-  Training seems to worsen (brutally) after ~40M steps (at least for lift policy).