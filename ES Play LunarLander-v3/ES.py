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
    用于连续动作空间的神经网络
    结构: MLP + 双输出层（均值+方差）
    '''
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        # 共享的特征提取层
        self.mlp = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh())
        
        # 均值输出层
        self.mean_l = nn.Linear(32, n_actions)
        self.mean_l.weight.data.mul_(0.1)  # 初始化缩小参数
        
        # 方差输出层（实际代码中未直接使用）
        self.var_l = nn.Linear(32, n_actions)
        self.var_l.weight.data.mul_(0.1)
        
        # 对数标准差参数（演化策略中未直接使用）
        self.logstd = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        # 前向传播：返回tanh激活后的动作均值
        ot_n = self.mlp(x.float())
        return torch.tanh(self.mean_l(ot_n))

def sample_noise(neural_net):
    '''
    为神经网络的每个参数生成高斯噪声
    返回: 噪声列表（与网络参数一一对应）
    '''
    return [np.random.normal(size=p.data.shape) for p in neural_net.parameters()]

def evaluate_neuralnet(nn, env):
    '''
    评估神经网络在环境中的表现
    返回: 单次episode的总奖励
    '''
    obs, _ = env.reset()
    total_reward = 0

    while True:
        with torch.no_grad():
            action = nn(torch.tensor(obs)).cpu().numpy()
        # 将动作裁剪到[-1, 1]范围
        clipped_action = np.clip(action, -1, 1)
        
        # Gymnasium返回5个值，需要处理terminated和truncated
        new_obs, reward, terminated, truncated, _ = env.step(clipped_action)
        total_reward += reward
        obs = new_obs
        
        if terminated or truncated:
            break
            
    return total_reward

def evaluate_noisy_net(noise, neural_net, env):
    '''
    评估添加噪声后的神经网络性能
    返回: 扰动后的网络在环境中的奖励
    '''
    original_params = neural_net.state_dict()
    
    # 给网络参数添加噪声
    for p, n in zip(neural_net.parameters(), noise):
        p.data += torch.FloatTensor(n * STD_NOISE)
    
    reward = evaluate_neuralnet(neural_net, env)
    neural_net.load_state_dict(original_params)  # 恢复原始参数
    return reward

def worker(params_queue, output_queue):
    '''
    工作进程函数: 持续接收参数，评估噪声扰动后的网络性能
    '''
    env = gym.make(ENV_NAME, continuous=True)  # 创建连续动作版本的环境
    actor = NeuralNetwork(env.observation_space.shape[0], env.action_space.shape[0])
    
    while True:
        params = params_queue.get()
        if params is None:  # 终止信号
            break
            
        actor.load_state_dict(params)
        seed = np.random.randint(1e6) # 生成随机种子
        np.random.seed(seed) # 确保噪声的一致性
        
        noise = sample_noise(actor)
        # 对称采样评估(正负噪声)
        pos_reward = evaluate_noisy_net(noise, actor, env)
        neg_reward = evaluate_noisy_net([-n for n in noise], actor, env)  # 对噪声列表中的每个元素取反
        
        output_queue.put([[pos_reward, neg_reward], seed])

def normalized_rank(rewards):
    '''
    对奖励进行排名并归一化到[-0.5, 0.5]范围
    '''
    ranked = ss.rankdata(rewards)
    return (ranked - 1) / (len(ranked) - 1) - 0.5

# 超参数配置
ENV_NAME = 'LunarLander-v3'          # Gymnasium环境名称
STD_NOISE = 0.05                     # 噪声标准差
BATCH_SIZE = 100                     # 每批采样数
LEARNING_RATE = 0.01                 # 学习率
MAX_ITERATIONS = 2000               # 最大训练迭代次数
MAX_WORKERS = 20                     # 并行工作进程数
VIDEOS_INTERVAL = 100                # 视频记录间隔
save_video_test = True

if __name__ == '__main__':
    # 设置多进程启动方法（确保与操作系统兼容）
    mp.set_start_method('spawn')
    
    # 初始化训练环境
    test_env = gym.make(ENV_NAME, continuous=True)
    input_shape = test_env.observation_space.shape[0] # 观测空间的维度
    action_shape = test_env.action_space.shape[0]     # 动作空间的维度
    
    # 初始化神经网络和优化器
    actor = NeuralNetwork(input_shape, action_shape)
    optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    
    # 创建用于视频记录的环境
    if save_video_test:
        writer_name = ENV_NAME
        test_env = gym.make(ENV_NAME, continuous=True, render_mode='rgb_array')
        test_env = RecordVideo(
            test_env, 
            video_folder=f"VIDEOS/TEST_VIDEOS_{writer_name}",
            episode_trigger=lambda x: True  # 记录所有episode
        )
    
    # 创建进程间通信队列
    params_queue = mp.Queue() # 用于将网络参数传递给工作进程
    output_queue = mp.Queue() # 用于从工作进程接收评估结果
    
    # 创建工作进程
    processes = []
    for _ in range(MAX_WORKERS):
        p = mp.Process(target=worker, args=(params_queue, output_queue))
        p.start()
        processes.append(p)

    reward_history = []  # 用于记录每轮的平均奖励
    plt.ion() # 允许动态更新图像
    fig, ax = plt.subplots()
    line, = ax.plot(reward_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Convergence')
    plt.show()
    # 主训练循环
    for iteration in range(MAX_ITERATIONS):
        start_time = time.time()
        
        # 将当前参数推送到队列供工作进程使用
        for _ in range(BATCH_SIZE):
            # 每个工作进程会从队列中获取参数，并评估噪声扰动后的网络性能
            params_queue.put(actor.state_dict())
        
        # 收集结果
        batch_rewards = []
        batch_noises = []
        for _ in range(BATCH_SIZE):
            # 每个工作进程返回一对奖励（正噪声和负噪声的奖励）和随机种子
            (pos_reward, neg_reward), seed = output_queue.get()
            np.random.seed(seed) # 使用种子确保噪声的一致性
            
            noise = sample_noise(actor) # 为每个参数生成高斯噪声
            batch_noises.extend([noise, [-n for n in noise]])  # 存储正负噪声
            batch_rewards.extend([pos_reward, neg_reward]) # 存储对应的奖励
        
        # 奖励归一化处理
        normalized_rewards = normalized_rank(batch_rewards)
        
        # 参数更新
        optimizer.zero_grad() # 清空之前的梯度
        for param, noise in zip(actor.parameters(), zip(*batch_noises)):
            # 计算梯度
            grad = sum([r * n for r, n in zip(normalized_rewards, noise)]) 
            grad = -grad / (BATCH_SIZE * STD_NOISE)  # 归一化并取负（梯度上升）
            param.grad = torch.FloatTensor(grad) # 将计算的梯度赋值给参数
        
        optimizer.step() # 更新网络参数
        
        # 记录训练指标
        avg_reward = np.mean(batch_rewards)
        reward_history.append(avg_reward)  # 记录每轮的奖励
        print(f"Iter: {iteration}, Reward: {avg_reward:.1f}, Time: {time.time()-start_time:.1f}s")
        
        # 定期测试并记录视频
        if iteration % VIDEOS_INTERVAL == 0:
            test_reward = evaluate_neuralnet(actor, test_env) # 评估网络在测试环境中的表现
            print(f"Test Reward: {test_reward}")
        

        # 在训练循环中更新图像
        line.set_ydata(reward_history)
        line.set_xdata(range(len(reward_history)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        # if iteration > 500 and np.mean(reward_history[-100:]) < np.mean(reward_history[-200:-100]):
        #     print("Early stopping due to no improvement.")
        #     break
    
    # 清理进程
    for _ in range(MAX_WORKERS):
        params_queue.put(None) # 向 params_queue 放入 None，作为终止信号
    for p in processes:
        p.join() # 等待所有工作进程结束

    # 训练结束后绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label='Average Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Reward Convergence Over Training Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_convergence.png')  # 保存图像
    plt.show()  # 显示图像