# குகா யூபாட் 
நகரக்கூடியது கையாளுதல் ah ஓட வைக்கிறோம் :) 

ED5125(Spr'25) திட்டத்திற்கான களஞ்சியம். இந்த திட்டம் வடமேற்கின் ME 449 கேப்ஸ்டோன் திட்டத்தை அடிப்படையாகக் கொண்டது. முழு திட்ட விளக்கத்தை காணலாம் [இங்கே](https://hades.mech.northwestern.edu/index.php/Mobile_Manipulation_Capstone_2023). 

## கண்ணோட்டம் 
1. இந்த களஞ்சியத்தில் CoppeliaSim இல் Kuka youBot மொபைல் கையாளுநர் (நான்கு மெகனம் சக்கரங்கள் மற்றும் 5R ரோபோ கை கொண்ட மொபைல் தளம்) பயன்படுத்தி பிக்-அண்ட்-பிளேஸ் பணியை உருவகப்படுத்துவதற்கான குறியீடு உள்ளது.
2. நிலையான தடைகள் சுவர் மற்றும் பழமையான வடிவங்களை உள்ளடக்கியது
3. பின்னூட்டக் கட்டுப்பாட்டாளர்
4. இயக்க திட்டமிடல்

## தொகுப்பு விளக்கம் 
இந்த திட்டம் occupancy_grid, base_planner, manipulator_planner, கட்டுப்படுத்தி simulation_manager பைதான் தொகுதிகள் (மைல்கற்கள்) என தொகுக்கப்பட்டுள்ளது: 

## அமைப்பு 
1. உங்கள் VENV இல் ZMQ API ஐ நிறுவ இந்த கட்டளையை இயக்கவும்- 'pip install coppeliasim-zmqremoteapi-client'
2. 2. simulation_manager.py இயக்கவும்

# Kuka youBot <EN>
Let's MoveIt Mobile Manipulator :)

Repository for ED5125 project (Spr'25). This project is based on Kuka YouBot mobile Manipultor performing motion planning.

## Overview
1. This repository contains code to simulate a pick-and-place task using a Kuka youBot mobile manipulator (a mobile base with four mecanum wheels and a 5R robot arm) in CoppeliaSim. 
2. Static Obstacles with wall and primitive shapes
3. Feedback Controller
4. Motion planning 

## Package Description
This project is packaged into occupancy_grid, base_planner, manipulator_planner, controller, simulation_manager Python modules (milestones): <br>

## Setup
1. run this command to install zmq api in your venv- `pip install coppeliasim-zmqremoteapi-client`
2. run simulation_manager.py

## Results

1. A* path of base with Euclidean heuristic
<img src="https://github.com/user-attachments/assets/0155c41a-d8ee-491d-87b1-1ba31527687a" alt="Alt Text" style="width:50%; height:auto;">

