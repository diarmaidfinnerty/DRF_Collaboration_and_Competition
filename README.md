# DRF_Collaboration_and_Competirion
Using DQN to Train an agent to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. 

## Project Details
In this project, double-jointed arms can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
The version of the environment contains 20 identical agents, each with its own copy of the environment. This allows for distributed learning. 

The environment is considered solved when the average reward over 100 episodes is at least +30.

## Installation
In order to run these scripts you will require
1. Follow instructions from this [link](https://github.com/udacity/deep-reinforcement-learning#dependencies) to ensure all dependencies are installed. 
2. Ensure that you execute *_cell 1_* of the `Naviagation.ipynb` notebook. `pip -q install ./python`.


# Setup
## Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

## Running the Code
1. Git clone the repository
2. Ensure that all steps in the Setup section have been completed
3. Run the `Tennis.ipynb` notebook. 
