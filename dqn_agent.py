# dqn_agent.py
"""实现DQN智能体"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Optional
from config import (
    GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, 
    LEARNING_RATE, BATCH_SIZE, TARGET_UPDATE_FREQUENCY,
    REPLAY_BUFFER_CAPACITY
)
from dqn import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, 
                 state_shape: Tuple[int, int, int], 
                 n_actions: int,
                 gamma: float = GAMMA, 
                 epsilon: float = EPSILON_START, 
                 epsilon_min: float = EPSILON_MIN, 
                 epsilon_decay: float = EPSILON_DECAY,
                 lr: float = LEARNING_RATE, 
                 batch_size: int = BATCH_SIZE, 
                 buffer_capacity: int = REPLAY_BUFFER_CAPACITY, 
                 target_update: int = TARGET_UPDATE_FREQUENCY):
        """
        初始化DQN智能体
        
        参数:
            state_shape: 状态空间形状
            n_actions: 动作数量
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
            lr: 学习率
            batch_size: 训练批量大小
            buffer_capacity: 经验回放缓冲区容量
            target_update: 目标网络更新频率
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 设备选择 (GPU优先)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 网络初始化
        self.policy_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net = DQN(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不更新梯度
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练计数器
        self.steps_done = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态
            
        返回:
            int: 选择的动作
        """
        # 探索-利用权衡
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)  # 随机探索
        
        # 利用策略网络选择最优动作
        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # 增加批量维度
            
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_step(self) -> Optional[float]:
        """
        执行一步训练
        
        返回:
            Optional[float]: 损失值，如果没有训练则为None
        """
        # 如果缓冲区不足，跳过
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 从缓冲区采样
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones = batch
        
        # 转换为张量
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)
        
        # 计算损失
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        
        self.optimizer.step()
        
        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"更新目标网络 (第 {self.steps_done} 步)")
        
        return loss.item()
    
    def remember(self, 
                 state: np.ndarray, 
                 action: int, 
                 reward: float, 
                 next_state: np.ndarray, 
                 done: bool):
        """
        保存经验到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def save_model(self, path: str):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        print(f"模型保存到: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint.get('steps_done', 0)
        print(f"从 {path} 加载模型")