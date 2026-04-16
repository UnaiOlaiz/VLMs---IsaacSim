I have designed 2 controllers, one for each different robot used in the workflow. One for the **Franka** arm and another for the **Jetbot**. 

1.  The Franka arm starts from the already implemented controller **RMPFlowController**. This one automatically avoids obstacles and also avoids crashing againts its structure. And I have coded this functions: 
-  **move_to_cube_top**: this functions receives the (x,y,z) target coordinates and moves there, it has a binary variable **keep_gripper_closed** to open/close the grippers, I will use it in the main pipeline.  And it stops until reaching a safe arbitrary distance variable. 
-  And the main **execute_movement** function that will orchestrate the whole movement and will be called in the main pipeline.
1. The Jetbot controller class was made from scratch, directly designing function that call the parts of the asset (chassis, wheels, joints, ...). Plus functions to stop the car, save the robot state (see below) and the main function. 
-  It basically works rotating first to the direction to face, and then go forward (correcting the direction at each step) until getting to the end position. The velocity reduction is not abrupt to not drop the objects (platform and cube) that will be holding in the real case. 
```python
	state_data = {

		"chassis_position": state["chassis_position"],

		"chassis_orientation": state["chassis_orientation"],

		"target_position": np.array(target_pos).tolist(),

}```
