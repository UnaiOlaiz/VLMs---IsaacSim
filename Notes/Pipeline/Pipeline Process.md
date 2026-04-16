[[BentoML Server]]
[[Control]]
[[Camera Capture]]

This is the main process of the pipeline, orchestrated by the *main* file which calls each needed component at the time (reduced to stupid basic, FOR NOW JUST WITH THE CONTROLLER):
![[Scenario.png]]
(This scenario is not the final one, it's just the most basic)

1. Get target cube and pallet color. 
2. Call **move_first_franka** function (**JUST FOR THE FRANKA IN THE RIGHT**). This function loads the detected prediction coordinates of the target cube and makes these movements:
	-  Move to the cube coordinates (x, y, SAFE HEIGHT OFFSET), that offset is usually the top of the cube. 
	-  Descent into the target (x, y, LOWER HEIGHT), this height is usually **height_cube/2**, so it stays in the middle. 
	-  Close grips, in the same coordinates as before, just change the value of that binary gripper variable.
	-  Lift the franka (with the cube between the grips) into a safe height. 
3. Call **place_on_jetbot** function to place the cube on top of the jetbot custom platform (the placing position is not deterministic). Basically go on top to a safe height, descent a little, open grips so the cube falls on top of the platform and finally lift the arm so it does not interfere with the next jetbot rotation+movement.
4. **move_jetbot** to move to the other franka. The position it stays are the predicted positions (with a little offset to don't crash with the frankas) by the VLM. It first rotates and goes forward until getting there, just as slow so the cube does not fall from the top. 
5. **pick_from_jetbot**: pick the cube from the jetbot. Same procedure, with safe heights. 
6. **place_on_palett**: load the palett predicted coordinates *JSON*, and place it there. 

And basically repeat it with whichever combination preferred.

![[FULL_PIPELINE.webm]]
This was the first approach, just worked with the red cube. All the cubes were in the default position. 

![[FULL_WITH_VARIATION.webm]]
This is the more advanced scenario (I need to make the execution faster xd):
- Red cube: default.
- Green cube: a little bit rotated. However, as the cube size is so small compared to the gripper opening size, it does not matter (*train RL agent with rotation randomization nevertheless*).
- Blue cube: On top of a custom platform to deal with different heights (harder work for the VLM detection).

Future updates/ideas: 
- Implement *RL* to the agents, until now I have been working with the controllers, but RL in *IsaacLab* is so problematic.
[[RL Progress]]