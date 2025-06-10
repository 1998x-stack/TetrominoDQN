# tetris_env.py
"""实现俄罗斯方块游戏环境，符合gym.Env接口"""
import numpy as np
import pygame
import gym
from gym import spaces
from typing import Tuple, Optional, List
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, 
    SIDEBAR_WIDTH, GAME_AREA_X, GAME_AREA_Y, SHAPES, SHAPE_COLORS, 
    BLACK, WHITE, RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW, ORANGE, 
    GRAY, LIGHT_GRAY, DARK_GRAY, BACKGROUND, GRID_COLOR, 
    FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_SMALL,
    INITIAL_FALL_SPEED, MIN_FALL_SPEED, LEVEL_SPEED_DECREMENT,
    LINES_PER_LEVEL, SCORE_MULTIPLIERS, MAX_STEPS_PER_EPISODE
)
from tetromino import Tetromino


class TetrisEnv(gym.Env):
    """俄罗斯方块游戏环境，符合gym.Env接口"""
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        初始化游戏环境
        
        参数:
            render_mode: 渲染模式，'human'表示使用pygame渲染
        """
        super().__init__()
        
        # 动作空间: 0=左移, 1=右移, 2=下移, 3=旋转, 4=直接下落, 5=无操作
        self.action_space = spaces.Discrete(6)
        
        # 状态空间: 两个通道 (固定方块, 当前方块)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32
        )
        
        # 渲染设置
        self.render_mode = render_mode
        if render_mode == 'human':
            self._initialize_pygame()
        else:
            self.screen = None
        
        # 游戏状态
        self.reset()
        
        # 游戏时钟
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont(None, FONT_SIZE_LARGE)
        self.font_medium = pygame.font.SysFont(None, FONT_SIZE_MEDIUM)
        self.font_small = pygame.font.SysFont(None, FONT_SIZE_SMALL)
    
    def _initialize_pygame(self):
        """初始化Pygame显示环境"""
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("俄罗斯方块 DQN 训练")
    
    def reset(self) -> np.ndarray:
        """
        重置游戏状态并返回初始观察值
        
        返回:
            np.ndarray: 初始状态观察值
        """
        # 游戏板: 0表示空，非0表示有方块
        self.board = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # 当前方块和下一个方块
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        
        # 游戏状态
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.fall_speed = INITIAL_FALL_SPEED
        self.fall_timer = 0
        self.steps = 0
        self.max_steps = MAX_STEPS_PER_EPISODE
        
        # 返回初始状态
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        获取当前游戏状态
        
        返回:
            np.ndarray: 状态数组 (2, GRID_HEIGHT, GRID_WIDTH)
            通道0: 固定方块
            通道1: 当前方块位置
        """
        # 创建两个通道的状态数组
        state = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        
        # 通道0: 固定方块
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.board[y][x] != 0:
                    state[0, y, x] = 1.0
        
        # 通道1: 当前方块位置
        if not self.game_over:
            for y, row in enumerate(self.current_piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        board_y = self.current_piece.y + y
                        board_x = self.current_piece.x + x
                        if 0 <= board_y < GRID_HEIGHT and 0 <= board_x < GRID_WIDTH:
                            state[1, board_y, board_x] = 1.0
        
        return state
    
    def valid_position(self, shape: List[List[int]], x: int, y: int) -> bool:
        """
        检查给定位置是否有效
        
        参数:
            shape: 方块形状
            x: x坐标
            y: y坐标
            
        返回:
            bool: 位置是否有效
        """
        # 检查位置是否有效
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    # 检查边界条件
                    board_y = y + row_idx
                    board_x = x + col_idx
                    
                    # 检查是否超出边界
                    if board_x < 0 or board_x >= GRID_WIDTH or board_y >= GRID_HEIGHT:
                        return False
                    
                    # 检查是否与已有方块重叠
                    if board_y >= 0 and self.board[board_y][board_x]:
                        return False
        
        return True
    
    def _place_piece(self):
        """将当前方块放置到游戏板上"""
        # 将当前方块放置到游戏板上
        for row_idx, row in enumerate(self.current_piece.shape):
            for col_idx, cell in enumerate(row):
                if cell and self.current_piece.y + row_idx >= 0:
                    self.board[self.current_piece.y + row_idx][self.current_piece.x + col_idx] = self.current_piece.color
        
        # 检查是否有完整的行
        self._clear_lines()
        
        # 生成新方块
        self.current_piece = self.next_piece
        self.next_piece = Tetromino()
        
        # 检查游戏是否结束
        if not self.valid_position(self.current_piece.shape, self.current_piece.x, self.current_piece.y):
            self.game_over = True
    
    def _clear_lines(self):
        """清除完整的行并更新分数"""
        lines_to_clear = []
        for row_idx in range(GRID_HEIGHT):
            if all(self.board[row_idx]):
                lines_to_clear.append(row_idx)
        
        # 清除行
        for row_idx in sorted(lines_to_clear, reverse=True):
            del self.board[row_idx]
            self.board.insert(0, [0 for _ in range(GRID_WIDTH)])
        
        # 更新分数
        if lines_to_clear:
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            
            # 计算消除的行数对应的分数
            line_multiplier = min(num_cleared, 4)
            self.score += SCORE_MULTIPLIERS[line_multiplier] * self.level
            
            # 更新等级和速度
            self.level = self.lines_cleared // LINES_PER_LEVEL + 1
            self.fall_speed = max(MIN_FALL_SPEED, INITIAL_FALL_SPEED - (self.level - 1) * LEVEL_SPEED_DECREMENT)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一个动作并返回结果
        
        参数:
            action: 要执行的动作
            
        返回:
            Tuple[np.ndarray, float, bool, dict]: 
                observation: 新的状态
                reward: 获得的奖励
                done: 是否结束
                info: 附加信息
        """
        if self.game_over:
            return self._get_state(), 0, True, {'reason': 'game_over'}
        
        # 记录初始分数用于计算奖励
        prev_score = self.score
        
        # 执行动作
        reward = 0
        
        if action == 0:  # 左移
            moved = self._move(-1, 0)
            reward = 0.1 if moved else -0.1
        elif action == 1:  # 右移
            moved = self._move(1, 0)
            reward = 0.1 if moved else -0.1
        elif action == 2:  # 下移
            moved = self._move(0, 1)
            reward = 0.2 if moved else -0.1
        elif action == 3:  # 旋转
            rotated = self._rotate_piece()
            reward = 0.3 if rotated else -0.1
        elif action == 4:  # 直接下落
            self._drop()
            # 直接下落给予更大奖励
            reward = 0.5
        
        # 自动下落
        self.fall_timer += 1/60  # 假设60FPS
        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            if not self._move(0, 1):
                self._place_piece()
        
        # 添加清除行的奖励
        reward += (self.score - prev_score) / 100.0
        
        # 游戏结束惩罚
        if self.game_over:
            reward -= 5.0
        
        # 检查步数限制
        self.steps += 1
        truncated = self.steps >= self.max_steps
        
        # 返回结果
        done = self.game_over or truncated
        info = {
            'score': self.score, 
            'lines': self.lines_cleared, 
            'level': self.level,
            'steps': self.steps,
            'truncated': truncated
        }
        
        return self._get_state(), reward, done, info
    
    def _move(self, dx: int, dy: int) -> bool:
        """
        移动当前方块
        
        参数:
            dx: x方向移动量
            dy: y方向移动量
            
        返回:
            bool: 移动是否成功
        """
        if self.game_over:
            return False
            
        return self.valid_position(
            self.current_piece.shape, 
            self.current_piece.x + dx, 
            self.current_piece.y + dy
        )
    
    def _rotate_piece(self) -> bool:
        """
        旋转当前方块
        
        返回:
            bool: 旋转是否成功
        """
        if self.game_over:
            return False
            
        rotated = self.current_piece.rotate()
        if self.valid_position(rotated, self.current_piece.x, self.current_piece.y):
            self.current_piece.shape = rotated
            return True
        return False
    
    def _drop(self):
        """直接下落当前方块"""
        if self.game_over:
            return
            
        # 创建当前方块的副本
        piece_copy = self.current_piece.copy()
        
        # 尝试下落到最低位置
        while self.valid_position(piece_copy.shape, piece_copy.x, piece_copy.y + 1):
            piece_copy.y += 1
        
        # 检查是否有移动
        if piece_copy.y > self.current_piece.y:
            self.current_piece = piece_copy
        
        # 放置方块
        self._place_piece()
    
    def render(self):
        """渲染当前游戏状态"""
        if self.render_mode != 'human':
            return
            
        # 绘制背景
        self.screen.fill(BACKGROUND)
        
        # 绘制游戏区域边框
        pygame.draw.rect(self.screen, LIGHT_GRAY, 
                         (GAME_AREA_X - 2, GAME_AREA_Y - 2, 
                          GRID_WIDTH * GRID_SIZE + 4, GRID_HEIGHT * GRID_SIZE + 4), 
                         2)
        
        # 绘制网格
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pygame.draw.rect(self.screen, GRID_COLOR, 
                                (GAME_AREA_X + x * GRID_SIZE, GAME_AREA_Y + y * GRID_SIZE, 
                                 GRID_SIZE, GRID_SIZE), 1)
        
        # 绘制已落下的方块
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.board[y][x]:
                    pygame.draw.rect(self.screen, self.board[y][x], 
                                   (GAME_AREA_X + x * GRID_SIZE, GAME_AREA_Y + y * GRID_SIZE, 
                                    GRID_SIZE, GRID_SIZE))
                    pygame.draw.rect(self.screen, WHITE, 
                                   (GAME_AREA_X + x * GRID_SIZE, GAME_AREA_Y + y * GRID_SIZE, 
                                    GRID_SIZE, GRID_SIZE), 1)
        
        # 绘制当前方块
        if not self.game_over:
            for y, row in enumerate(self.current_piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(self.screen, self.current_piece.color, 
                                       (GAME_AREA_X + (self.current_piece.x + x) * GRID_SIZE, 
                                        GAME_AREA_Y + (self.current_piece.y + y) * GRID_SIZE, 
                                        GRID_SIZE, GRID_SIZE))
                        pygame.draw.rect(self.screen, WHITE, 
                                       (GAME_AREA_X + (self.current_piece.x + x) * GRID_SIZE, 
                                        GAME_AREA_Y + (self.current_piece.y + y) * GRID_SIZE, 
                                        GRID_SIZE, GRID_SIZE), 1)
        
        # 绘制侧边栏
        sidebar_x = GAME_AREA_X + GRID_WIDTH * GRID_SIZE + 20
        
        # 绘制下一个方块预览
        next_text = self.font_large.render("下一个方块:", True, YELLOW)
        self.screen.blit(next_text, (sidebar_x, GAME_AREA_Y))
        
        # 绘制下一个方块
        preview_x = sidebar_x + 30
        preview_y = GAME_AREA_Y + 50
        for y, row in enumerate(self.next_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, self.next_piece.color, 
                                   (preview_x + x * GRID_SIZE, preview_y + y * GRID_SIZE, 
                                    GRID_SIZE, GRID_SIZE))
                    pygame.draw.rect(self.screen, WHITE, 
                                   (preview_x + x * GRID_SIZE, preview_y + y * GRID_SIZE, 
                                    GRID_SIZE, GRID_SIZE), 1)
        
        # 绘制分数
        score_text = self.font_large.render(f"分数: {self.score}", True, CYAN)
        self.screen.blit(score_text, (sidebar_x, preview_y + 120))
        
        # 绘制等级
        level_text = self.font_large.render(f"等级: {self.level}", True, GREEN)
        self.screen.blit(level_text, (sidebar_x, preview_y + 170))
        
        # 绘制已消除行数
        lines_text = self.font_large.render(f"消除行数: {self.lines_cleared}", True, MAGENTA)
        self.screen.blit(lines_text, (sidebar_x, preview_y + 220))
        
        # 绘制游戏标题
        title = pygame.font.SysFont(None, 48).render("俄罗斯方块 DQN", True, ORANGE)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))
        
        # 绘制操作说明
        controls_y = preview_y + 280
        controls = [
            "智能体控制:",
            "动作 0: 左移",
            "动作 1: 右移",
            "动作 2: 下移",
            "动作 3: 旋转",
            "动作 4: 直接下落",
            "动作 5: 无操作",
            "R : 重新开始",
            "ESC : 退出"
        ]
        
        for i, text in enumerate(controls):
            ctrl_text = self.font_small.render(text, True, LIGHT_GRAY)
            self.screen.blit(ctrl_text, (sidebar_x, controls_y + i * 30))
        
        # 绘制步数信息
        steps_text = self.font_medium.render(f"步数: {self.steps}/{MAX_STEPS_PER_EPISODE}", True, LIGHT_GRAY)
        self.screen.blit(steps_text, (sidebar_x, controls_y + len(controls) * 30 + 10))
        
        # 如果游戏结束，显示游戏结束文本
        if self.game_over:
            game_over_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            game_over_surface.set_alpha(180)
            game_over_surface.fill(BLACK)
            self.screen.blit(game_over_surface, (0, 0))
            
            game_over_text = self.font_large.render("游戏结束!", True, RED)
            restart_text = self.font_large.render("按 R 键重新开始", True, YELLOW)
            
            self.screen.blit(game_over_text, 
                       (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                        SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(restart_text, 
                       (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 
                        SCREEN_HEIGHT // 2 + 10))
        
        # 更新屏幕
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """关闭环境并释放资源"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()