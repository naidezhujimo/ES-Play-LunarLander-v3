import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch import optim

import scipy.stats as ss
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

class NeuralNetwork(nn.Module):
    '''
    A neural network for continuous action spaces.
    Structure: MLP + dual output layers (mean + variance).
    '''
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        # Shared feature extraction layer
        self.mlp = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh())
        
        # Mean output layer
        self.mean_l = nn.Linear(32, n_actions)
        self.mean_l.weight.data.mul_(0.1)  # Initialize parameters with smaller values
        
        # Variance output layer (not directly used in this code)
        self.var_l = nn.Linear(32, n_actions)
        self.var_l.weight.data.mul_(0.1)
        
        # Log standard deviation parameter (not directly used in evolution strategies)
        self.logstd = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        # Forward propagation: Return the action mean after Tanh activation
        ot_n = self.mlp(x.float())
        return torch.tanh(self.mean_l(ot_n))

def sample_noise(neural_net):
    '''
    Generate Gaussian noise for each parameter of the neural network.
    Return: A list of noise (corresponding to each network parameter).
    '''
    return [np.random.normal(size=p.data.shape) for p in neural_net.parameters()]

def evaluate_neuralnet(nn, env):
    '''
    Evaluate the performance of the neural network in the environment.
    Return: The total reward of a single episode.
    '''
    obs, _ = env.reset()
    total_reward = 0

    while True:
        with torch.no_grad():
            action = nn(torch.tensor(obs)).cpu().numpy()
        # Clip the action to the range [-1, 1]
        clipped_action = np.clip(action, -1, 1)
        
        # Gymnasium returns 5 values, need to handle terminated and truncated
        new_obs, reward, terminated, truncated, _ = env.step(clipped_action)
        total_reward += reward
        obs = new_obs
        
        if terminated or truncated:
            break
            
    return total_reward

def evaluate_noisy_net(noise, neural_net, env):
    '''
    Evaluate the performance of the neural network after adding noise.
    Return: The reward of the perturbed network in the environment.
    '''
    original_params = neural_net.state_dict()
    
    # Add noise to the network parameters
    for p, n in zip(neural_net.parameters(), noise):
        p.data += torch.FloatTensor(n * STD_NOISE)
    
    reward = evaluate_neuralnet(neural_net, env)
    neural_net.load_state_dict(original_params)  # Restore the original parameters
    return reward

def worker(params_queue, output_queue):
    '''
    Worker process function: Continuously receive parameters and evaluate the performance of the perturbed network.
    '''
    env = gym.make(ENV_NAME, continuous=True)  # Create a continuous action version of the environment
    actor = NeuralNetwork(env.observation_space.shape[0], env.action_space.shape[0])
    
    while True:
        params = params_queue.get()
        if params is None:  # Termination signal
            break
            
        actor.load_state_dict(params)
        seed = np.random.randint(1e6) # Generate a random seed
        np.random.seed(seed) # Ensure consistent noise
        
        noise = sample_noise(actor)
        # Symmetric sampling evaluation (positive and negative noise)
        pos_reward = evaluate_noisy_net(noise, actor, env)
        neg_reward = evaluate_noisy_net([-n for n in noise], actor, env)  # Take the opposite of each element in the noise list
        
        output_queue.put([[pos_reward, neg_reward], seed])

def normalized_rank(rewards):
    '''
    Rank the rewards and normalize them to the range [-0.5, 0.5].
    '''
    ranked = ss.rankdata(rewards)
    return (ranked - 1) / (len(ranked) - 1) - 0.5

# Hyperparameter configuration
ENV_NAME = 'LunarLander-v3'          # Environment name in Gymnasium
STD_NOISE = 0.05                     # Standard deviation of noise
BATCH_SIZE = 100                     # Number of samples per batch
LEARNING_RATE = 0.01                 # Learning rate
MAX_ITERATIONS = 2000               # Maximum number of training iterations
MAX_WORKERS = 20                     # Number of parallel worker processes
VIDEOS_INTERVAL = 100                # Interval for recording videos
save_video_test = True

if __name__ == '__main__':
    # Set the multiprocessing start method (ensure compatibility with the operating system)
    mp.set_start_method('spawn')
    
    # Initialize the training environment
    test_env = gym.make(ENV_NAME, continuous=True)
    input_shape = test_env.observation_space.shape[0] # Dimension of the observation space
    action_shape = test_env.action_space.shape[0]     # Dimension of the action space
    
    # Initialize the neural network and optimizer
    actor = NeuralNetwork(input_shape, action_shape)
    optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    
    # Create an environment for video recording
    if save_video_test:
        writer_name = ENV_NAME
        test_env = gym.make(ENV_NAME, continuous=True, render_mode='rgb_array')
        test_env = RecordVideo(
            test_env, 
            video_folder=f"VIDEOS/TEST_VIDEOS_{writer_name}",
            episode_trigger=lambda x: True  # Record all episodes
        )
    
    # Create queues for inter-process communication
    params_queue = mp.Queue() # For passing network parameters to worker processes
    output_queue = mp.Queue() # For receiving evaluation results from worker processes
    
    # Create worker processes
    processes = []
    for _ in range(MAX_WORKERS):
        p = mp.Process(target=worker, args=(params_queue, output_queue))
        p.start()
        processes.append(p)

    reward_history = []  # To record the average reward of each iteration
    plt.ion() # Enable dynamic image updates
    fig, ax = plt.subplots()
    line, = ax.plot(reward_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Convergence')
    plt.show()
    # Main training loop
    for iteration in range(MAX_ITERATIONS):
        start_time = time.time()
        
        # Push the current parameters to the queue for worker processes to use
        for _ in range(BATCH_SIZE):
            # Each worker process will get the parameters and evaluate the performance of the perturbed network
            params_queue.put(actor.state_dict())
        
        # Collect results
        batch_rewards = []
        batch_noises = []
        for _ in range(BATCH_SIZE):
            # Each worker process returns a pair of rewards (positive and negative noise rewards) and a random seed
            (pos_reward, neg_reward), seed = output_queue.get()
            np.random.seed(seed) # Use the seed to ensure consistent noise
            
            noise = sample_noise(actor) # Generate Gaussian noise for each parameter
            batch_noises.extend([noise, [-n for n in noise]])  # Store positive and negative noise
            batch_rewards.extend([pos_reward, neg_reward]) # Store the corresponding rewards
        
        # Normalize the rewards
        normalized_rewards = normalized_rank(batch_rewards)
        
        # Update parameters
        optimizer.zero_grad() # Clear previous gradients
        for param, noise in zip(actor.parameters(), zip(*batch_noises)):
            # Compute the gradient
            grad = sum([r * n for r, n in zip(normalized_rewards, noise)]) 
            grad = -grad / (BATCH_SIZE * STD_NOISE)  # Normalize and negate (gradient ascent)
            param.grad = torch.FloatTensor(grad) # Assign the computed gradient to the parameter
        
        optimizer.step() # Update network parameters
        
        # Record training metrics
        avg_reward = np.mean(batch_rewards)
        reward_history.append(avg_reward)  # Record the average reward of each iteration
        print(f"Iter: {iteration}, Reward: {avg_reward:.1f}, Time: {time.time()-start_time:.1f}s")
        
        # Periodically test and record videos
        if iteration % VIDEOS_INTERVAL == 0:
            test_reward = evaluate_neuralnet(actor, test_env) # Evaluate the network's performance in the test environment
            print(f"Test Reward: {test_reward}")
        

        # Update the image in the training loop
        line.set_ydata(reward_history)
        line.set_xdata(range(len(reward_history)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        # if iteration > 500 and np.mean(reward_history[-100:]) < np.mean(reward_history[-200:-100]):
        #     print("Early stopping due to no improvement.")
        #     break
    
    # Clean up processes
    for _ in range(MAX_WORKERS):
        params_queue.put(None) # Put None into params_queue as a termination signal
    for p in processes:
        p.join() # Wait for all worker processes to finish

    # Plot the reward curve after training
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label='Average Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Reward Convergence Over Training Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_convergence.png')  # Save the image
    plt.show()  # Display the image