# replay_buffer.py
"""实现经验回放缓冲区"""
import numpy as np
from collections import deque
from typing import Tuple, Optional


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """
        初始化缓冲区
        
        参数:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, 
             state: np.ndarray, 
             action: int, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool):
        """
        添加经验到缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Optional[Tuple]:
        """
        从缓冲区随机采样
        
        参数:
            batch_size: 批量大小
            
        返回:
            Optional[Tuple]: 批量的 (state, action, reward, next_state, done)
        """
        if len(self.buffer) < batch_size:
            return None
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """返回缓冲区大小"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return len(self.buffer) == self.capacity