# Reinforcement Learning with Evolution Strategies (ES) in LunarLander-v3

This repository contains an implementation of a reinforcement learning algorithm using Evolution Strategies (ES) to solve the `LunarLander-v3` environment from the Gymnasium library. The algorithm uses a neural network to parameterize the policy and optimizes it using noise perturbations and rank-based fitness shaping.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Code Structure](#code-structure)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

Evolution Strategies (ES) are a family of optimization algorithms inspired by natural evolution. They are particularly useful for optimizing policies in reinforcement learning, especially in environments with continuous action spaces. This project applies ES to the `LunarLander-v3` environment, where the goal is to land a lunar module safely on the moon's surface.

## Dependencies

To run this code, you need the following Python libraries:

- `numpy`
- `torch` (PyTorch)
- `gymnasium`
- `scipy`
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install numpy torch gymnasium scipy matplotlib
```

## Code Structure

The code is structured as follows:

- **Neural Network (`NeuralNetwork` class)**: A multi-layer perceptron (MLP) with two output heads for mean and variance (though variance is not directly used in this implementation).
- **Noise Sampling (`sample_noise` function)**: Generates Gaussian noise for perturbing the neural network parameters.
- **Evaluation Functions (`evaluate_neuralnet` and `evaluate_noisy_net`)**: Evaluate the performance of the neural network in the environment, with and without noise perturbations.
- **Worker Function (`worker`)**: A multiprocessing worker that evaluates noisy versions of the neural network in parallel.
- **Training Loop**: The main loop that iteratively updates the neural network parameters using the ES algorithm.

## How It Works

1. **Policy Parameterization**: The policy is parameterized by a neural network that outputs the mean action for a given state. The network is perturbed by adding Gaussian noise to its parameters.
2. **Noise Perturbation**: For each iteration, multiple noisy versions of the network are evaluated in the environment. The performance of each noisy network is recorded.
3. **Rank-Based Fitness Shaping**: The rewards obtained from the noisy evaluations are ranked and normalized. This ranking is used to compute the gradient for updating the network parameters.
4. **Parameter Update**: The network parameters are updated using gradient ascent, where the gradient is computed based on the performance of the noisy networks.
5. **Parallelization**: The evaluation of noisy networks is parallelized using multiple worker processes, which significantly speeds up the training process.

## Usage

To run the training script, simply execute the following command:

```bash
python es_lunarlander.py
```

### Key Hyperparameters

- `ENV_NAME`: The Gymnasium environment name (`LunarLander-v3`).
- `STD_NOISE`: The standard deviation of the Gaussian noise used for perturbations.
- `BATCH_SIZE`: The number of noisy evaluations per iteration.
- `LEARNING_RATE`: The learning rate for the Adam optimizer.
- `MAX_ITERATIONS`: The maximum number of training iterations.
- `MAX_WORKERS`: The number of parallel worker processes.

### Video Recording

The code includes functionality to record videos of the agent's performance at regular intervals. Videos are saved in the `VIDEOS/TEST_VIDEOS_LunarLander-v3` directory.

## Results

The training progress is visualized using a plot of the average reward over iterations. The plot is updated dynamically during training and saved as `reward_convergence.png` at the end of the training process.

### Example Output

```plaintext
Iter: 0, Reward: -150.3, Time: 5.2s
Iter: 1, Reward: -120.5, Time: 5.1s
...
Iter: 1999, Reward: 250.7, Time: 5.0s
Test Reward: 280.3
```

The visualisation data is available in folders.
