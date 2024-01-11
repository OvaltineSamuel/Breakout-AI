# Breakout-AI w/ DQN

Welcome to BreakoutAI - an AI-powered agent mastering the Atari Breakout game using reinforcement learning!

![](https://github.com/OvaltineSamuel/Breakout-AI/blob/main/Results/FootageofAIBreakout.gif)

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


## Training Results
After extensive training and experimentation, improvements were achieved by addressing initial challenges, such as a low buffer size. Hyperparameter adjustments, CUDA cores utilization, and exploration rate changes led to a significant increase in mean episode reward, demonstrating the model's enhanced performance.

In the results plot presented below, a clear distinction emerges between the early versions (ver.01-05) and the optimized models (ver.06-11). The initial models, characterized as unoptimized, exhibit lower performance in terms of episode reward mean. Contrastingly, the optimized models, a product of parameter adjustments, demonstrate significantly improved performance. This enhancement underscores the crucial impact of parameter tuning on the overall effectiveness of the models, with the latter versions showcasing superior capabilities and higher episode reward means.

![](https://github.com/OvaltineSamuel/Breakout-AI/blob/main/Results/Result%20Plot%201%20(All%20models).png)

Watch full project result demotration video [here](https://drive.google.com/file/d/1i4fBTI0rRbnVcMY1ALyNLsVfxf8G97SL/view).


## Project Conclusion
In this project, our primary objective was to explore the capabilities of Deep Q-Networks (DQN) in training an AI agent to excel at Breakout, drawing inspiration from the Atari Deepmind Paper. Throughout all of this, it has shed light on the profound abilities of DQN and reinforcement learning algorithms within artificial intelligence (AI) through the context of video games. This exploration of AI and gaming has advanced the boundaries of machine learning, bridging the gap towards human-like thinking and decision-making.

After days of extensive training and analysis, our AI agent has demonstrated problem-solving abilities within the dynamic environment of Breakout, emphasizing spatial awareness, strategic planning, and precision–qualities we aim to instill in our agent. In achieving our goal, we’ve not only explored the abilities of DQN and RL algorithms, but also contributed to the understanding of machine learning, ultimately pushing the advancement towards human-like AI decision-making.

Read more detail result and full project report [here](https://github.com/OvaltineSamuel/Breakout-AI/blob/main/Project%20Report%20(DQN-Breakout).pdf).