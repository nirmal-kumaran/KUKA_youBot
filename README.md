# Kuka youBot 
மொபைல் கையாளுதல் ah ஓட வைக்கிறோம் :) 

ED5125 திட்டத்திற்கான களஞ்சியம் (Spr'25). இந்த திட்டம் வடமேற்கின் ME 449 கேப்ஸ்டோன் திட்டத்தை அடிப்படையாகக் கொண்டது. அவர்களுக்கே கிரெடிட்ஸ். முழு திட்ட விளக்கத்தை காணலாம் [இங்கே]https://hades.mech.northwestern.edu/index.php/Mobile_Manipulation_Capstone_2023). 

## கண்ணோட்டம் 
1. இந்த களஞ்சியத்தில் CoppeliaSim இல் Kuka youBot மொபைல் கையாளுநர் (நான்கு மெகனம் சக்கரங்கள் மற்றும் 5R ரோபோ கை கொண்ட மொபைல் தளம்) பயன்படுத்தி பிக்-அண்ட்-பிளேஸ் பணியை உருவகப்படுத்துவதற்கான குறியீடு உள்ளது.
2. நிலையான தடைகள் சுவர் மற்றும் பழமையான வடிவங்களை உள்ளடக்கியது
3. கருத்துக் கட்டுப்பாட்டாளர்
4. இயக்க திட்டமிடல்

<EN>
# Kuka youBot Mobile Manipulation
Let's MoveIt Mobile Manipulator :)

Repository for ED5125 project (Spr'25). This project is based on Northwestern's ME 449 capstone project. Credits to them. Full project description can be found [here](https://hades.mech.northwestern.edu/index.php/Mobile_Manipulation_Capstone_2023).

## Overview
1. This repository contains code to simulate a pick-and-place task using a Kuka youBot mobile manipulator (a mobile base with four mecanum wheels and a 5R robot arm) in CoppeliaSim. 
2. Static Obstacles include wall and primitive shapes
3. Feedback Controller
4. Motion planning 

## Package Description
This project is packaged into 3 Python modules (milestones): <br>

1. `next_state.py` - Milestone 1 code. The function `NextState()` computes the configuration of the robot for the next time step using first-order Euler integration.<br>
2. `trajectory_generator.py` - Milestone 2 code. The function `TrajectoryGenerator()` generates the reference trajectory for the robot's end effector.<br>
3. `feedback_control.py` - Milestone 3 code. The function `FeedbackControl()` uses kinematic task-space feedforward plus feedback control to continuously reduce deviations from the reference trajectory.<br>
4. `mobile_manipulation.py` - Full Program Implementation. Uses the functions defined in the 3 milestone modules to generate the robot trajectory.
