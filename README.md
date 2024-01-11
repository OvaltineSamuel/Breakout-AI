# Breakout-AI w/ DQN

Welcome to BreakoutAI - an AI-powered agent mastering the Atari Breakout game using reinforcement learning!

## Overview

BreakoutAI is a project aimed at developing a robust AI model capable of learning and excelling at playing the classic Atari Breakout game. The project leverages the Stable Baselines3 library for reinforcement learning and deep Q-networks to train the agent.

## Features

- **Reinforcement Learning:** BreakoutAI uses the DQN (Deep Q-Network) algorithm for training the agent.
- **OpenAI Gym Integration:** The project utilizes the OpenAI Gym framework, providing a standardized environment for training and testing reinforcement learning models.
- **Hyperparameter Tuning:** BreakoutAI includes configurations for various hyperparameters, allowing users to experiment with and optimize the model's performance.

## Getting Started

Follow these steps to get BreakoutAI up and running:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/OvaltineSamuel/Breakout-AI.git
    cd BreakoutAI

2. **Install Dependencies using Conda:**
    
    install in a new conda environment with:
    ````
    conda create -n YOURENVNAME python==3.8

    conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch

    pip install stable-baselines3[extra]
    ````
    for use with Jupyter notebooks:
    ````
    pip install ipykernel
    ````
    for progress bar:
    ````
    pip install ipywidgets
    ````
    for video recording
    ````
    pip install moviepy
    ````
    for logging with wandb
    ````
    pip install wandb
    ````

3. **Train the model**

    you can simply train the model in the notebook (BreakoutStableBaselines.ipynb).

## Configuration

Adjust the configuration parameters in the training script (BreakoutStableBaselines.ipynb) to experiment with different settings and improve model performance.


## Results

