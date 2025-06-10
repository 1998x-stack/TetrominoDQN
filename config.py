"""
俄罗斯方块游戏 + DQN 智能体训练系统

文件结构:
- tetris_env.py: 游戏环境实现
- tetromino.py: 方块类
- dqn.py: DQN网络模型
- replay_buffer.py: 经验回放缓冲区
- dqn_agent.py: DQN智能体实现
- train.py: 训练和测试主程序
- config.py: 配置常量

实现遵循 Google 风格指南，并添加了详细的中文注释
"""

# config.py
"""包含游戏和训练的所有配置常量"""
import pygame

# Pygame 初始化
pygame.init()

# 屏幕尺寸
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# 游戏网格设置
GRID_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SIDEBAR_WIDTH = 200

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (40, 40, 40)
LIGHT_GRAY = (100, 100, 100)
DARK_GRAY = (20, 20, 20)
BACKGROUND = (15, 15, 30)
GRID_COLOR = (30, 30, 60)

# 游戏区域位置
GAME_AREA_X = (SCREEN_WIDTH - SIDEBAR_WIDTH - GRID_WIDTH * GRID_SIZE) // 2
GAME_AREA_Y = (SCREEN_HEIGHT - GRID_HEIGHT * GRID_SIZE) // 2

# 方块形状定义
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[1, 1], [1, 1]],  # O
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]]  # Z
]

# 方块颜色
SHAPE_COLORS = [CYAN, MAGENTA, ORANGE, BLUE, YELLOW, GREEN, RED]

# DQN 训练参数
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
REPLAY_BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQUENCY = 1000
MAX_STEPS_PER_EPISODE = 2000
NUM_EPISODES = 1000
RENDER_FREQUENCY = 50

# 游戏速度参数
INITIAL_FALL_SPEED = 0.5
MIN_FALL_SPEED = 0.05
LEVEL_SPEED_DECREMENT = 0.05
LINES_PER_LEVEL = 10
SCORE_MULTIPLIERS = {1: 100, 2: 300, 3: 500, 4: 800}

# 字体设置
FONT_SIZE_LARGE = 36
FONT_SIZE_MEDIUM = 28
FONT_SIZE_SMALL = 24