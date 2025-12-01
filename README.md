# Kuka youBot <EN>
Let's MoveIt Mobile Manipulator :)

This project is based on Kuka YouBot mobile Manipultor performing motion planning.

## Overview
1. This repository contains code to simulate a pick-and-place task using a Kuka youBot mobile manipulator (a mobile base with four mecanum wheels and a 5R robot arm) in CoppeliaSim. 
2. Static Obstacles with wall and primitive shapes
3. Feedback Controller
4. Motion planning 

## Package Description
This project is packaged into occupancy_grid, base_planner, manipulator_planner, controller, simulation_manager

## Setup
1. run this command to install zmq api in your venv- `pip install coppeliasim-zmqremoteapi-client`
2. run simulation_manager.py

## Results
1. A* path of base with Euclidean heuristic
<img src="https://github.com/user-attachments/assets/0155c41a-d8ee-491d-87b1-1ba31527687a" alt="Alt Text" style="width:50%; height:auto;">

<img src="https://github.com/user-attachments/assets/08be98e5-804d-4bc2-8a96-945d76604ccd" alt="Alt Text" style="width:50%; height:auto;">

<img src="./scene/0.png" alt="Alt Text" style="width:50%; height:auto;">

<img src="./scene/0(1).png" alt="Alt Text" style="width:50%; height:auto;">

<img src="./scene/0(2).png" alt="Alt Text" style="width:50%; height:auto;">
