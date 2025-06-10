# dqn.py
"""实现深度Q网络模型"""
import torch
import torch.nn as nn
from typing import Tuple


class DQN(nn.Module):
    """深度Q网络 (DQN)"""
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        """
        初始化DQN网络
        
        参数:
            input_shape: 输入状态形状 (channels, height, width)
            n_actions: 动作数量
        """
        super().__init__()
        
        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 计算卷积层输出大小
        conv_out_size = self._get_conv_out(input_shape)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    
    def _get_conv_out(self, shape: Tuple[int, int, int]) -> int:
        """
        计算卷积层输出大小
        
        参数:
            shape: 输入形状 (channels, height, width)
            
        返回:
            int: 卷积层输出大小
        """
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            out = self.conv(x)
            return int(torch.prod(torch.tensor(out.size()[1:])))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入状态
            
        返回:
            torch.Tensor: 动作价值
        """
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)