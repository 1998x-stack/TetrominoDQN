# tetromino.py
"""实现俄罗斯方块的单个方块类"""
import random
from typing import List, Tuple
from config import SHAPES, SHAPE_COLORS, GRID_WIDTH


class Tetromino:
    """表示俄罗斯方块中的一个方块"""
    
    def __init__(self, shape_index = None):
        """
        初始化方块
        
        参数:
            shape_index: 指定方块形状的索引 (可选)
        """
        if shape_index is None:
            self.shape_index = random.randint(0, len(SHAPES) - 1)
        else:
            self.shape_index = shape_index % len(SHAPES)
        
        self.shape = SHAPES[self.shape_index]
        self.color = SHAPE_COLORS[self.shape_index]
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
    
    def rotate(self) -> List[List[int]]:
        """
        旋转方块并返回旋转后的形状
        
        返回:
            List[List[int]]: 旋转后的方块形状
        """
        # 旋转方块 (转置然后反转每一行)
        rows = len(self.shape)
        cols = len(self.shape[0])
        rotated = [[0] * rows for _ in range(cols)]
        
        for r in range(rows):
            for c in range(cols):
                rotated[c][rows - 1 - r] = self.shape[r][c]
        
        return rotated
    
    def get_dimensions(self) -> Tuple[int, int]:
        """
        获取方块的尺寸
        
        返回:
            Tuple[int, int]: (高度, 宽度)
        """
        return len(self.shape), len(self.shape[0])
    
    def copy(self):
        """创建方块的副本"""
        new_tetromino = Tetromino(self.shape_index)
        new_tetromino.shape = [row[:] for row in self.shape]
        new_tetromino.x = self.x
        new_tetromino.y = self.y
        return new_tetromino