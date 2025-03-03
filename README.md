# Kuka youBot Mobile Manipulation
Mobile Manipulator ah ஓட வைக்கிறோம் :)

Repository for ED5125 project (Spr'25). This project is based on Northwestern's ME 449 capstone project. Credits to them. Full project description can be found [here](https://hades.mech.northwestern.edu/index.php/Mobile_Manipulation_Capstone_2023).

## Overview
1. This repository contains code to simulate a pick-and-place task using a Kuka youBot mobile manipulator (a mobile base with four mecanum wheels and a 5R robot arm) in CoppeliaSim. 
2. Static Obstacles include wall and primitive shapes
3. 
4. Motion planning 

## Package Description
This project is packaged into 3 Python modules (milestones): <br>

1. `next_state.py` - Milestone 1 code. The function `NextState()` computes the configuration of the robot for the next time step using first-order Euler integration.<br>
2. `trajectory_generator.py` - Milestone 2 code. The function `TrajectoryGenerator()` generates the reference trajectory for the robot's end effector.<br>
3. `feedback_control.py` - Milestone 3 code. The function `FeedbackControl()` uses kinematic task-space feedforward plus feedback control to continuously reduce deviations from the reference trajectory.<br>
4. `mobile_manipulation.py` - Full Program Implementation. Uses the functions defined in the 3 milestone modules to generate the robot trajectory.
