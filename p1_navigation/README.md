# Navigation Project

## Project Details

This project trains an agent to navigate a large square world and collect yellow bananas while avoiding blue bananas using Deep Q-Network (DQN).

- **State space**: 37 dimensions containing the agent's velocity and ray-based perception of objects around the agent's forward direction.
- **Action space**: 4 discrete actions:
  - `0` - move forward
  - `1` - move backward
  - `2` - turn left
  - `3` - turn right
- **Reward**: +1 for collecting a yellow banana, -1 for collecting a blue banana
- **Solved**: The environment is considered solved when the agent gets an average score of +13 over 100 consecutive episodes.

## Getting Started

### Dependencies

1. Python 3.6+
2. PyTorch (with CUDA support recommended)
3. Unity ML-Agents (unityagents)
4. NumPy
5. Matplotlib

### Installation

```
pip install torch numpy matplotlib
```

For the Unity environment, follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.

Download the Banana environment:
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

Place the unzipped file in the `p1_navigation/` folder.

## Instructions

To train the agent, open `Navigation.ipynb` in Jupyter Notebook and run all cells. The training will:

1. Initialize the Banana environment
2. Create a DQN agent
3. Train until the average score reaches +13 over 100 episodes
4. Save trained weights to `checkpoint.pth`
5. Plot the training scores

## Files

- `Navigation.ipynb` - Main notebook for training and evaluation
- `dqn_agent.py` - DQN Agent and Replay Buffer implementation
- `model.py` - Neural network architecture (QNetwork)
- `checkpoint.pth` - Saved model weights (created after training)
- `Report.md` - Detailed report of the implementation
