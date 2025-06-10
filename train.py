# train.py
"""训练和测试主程序"""
import pygame
import os
import time
import numpy as np
from typing import Tuple
from config import (
    NUM_EPISODES, RENDER_FREQUENCY, 
    REPLAY_BUFFER_CAPACITY, LEARNING_RATE
)
from tetris_env import TetrisEnv
from dqn_agent import DQNAgent
from loguru import logger
from tensorboardX import SummaryWriter

def setup_tensorboard():
    """设置TensorBoard"""
    writer = SummaryWriter(log_dir="runs", flush_secs=10)
    logger.info("TensorBoard已启动")
    return writer

def setup_environment():
    """设置游戏环境"""
    env = TetrisEnv()
    logger.info("游戏环境已初始化")
    return env


def ensure_dir(path: str):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def setup_logging():
    """设置日志记录"""
    log_dir = "logs"
    ensure_dir(log_dir)
    log_file = os.path.join(log_dir, "training.log")
    logger.add(log_file, rotation="1 MB", level="INFO")
    logger.info("日志记录已启动")

def train_agent(env: TetrisEnv, agent: DQNAgent, num_episodes: int = NUM_EPISODES) -> Tuple:
    """
    训练智能体
    
    参数:
        env: 游戏环境
        agent: DQN智能体
        num_episodes: 训练轮数
        
    返回:
        Tuple: (分数列表, 损失列表)
    """
    scores = []
    all_losses = []
    best_score = -float('inf')
    start_time = time.time()
    summary_writer = setup_tensorboard()
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []
        step_count = 0
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 训练
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            # 更新状态
            state = next_state
            step_count += 1
            
            # 渲染
            if RENDER_FREQUENCY > 0 and episode % RENDER_FREQUENCY == 0:
                env.render()
        
        # 更新探索率
        agent.update_epsilon()
        
        # 计算平均损失
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        # 记录结果
        scores.append(total_reward)
        all_losses.append(avg_loss)
        
        # 更新最佳模型
        if total_reward > best_score:
            best_score = total_reward
            agent.save_model("models/best_model.pth")
        
        # 打印训练信息
        elapsed_time = time.time() - start_time
        summary_writer.add_scalar('Total Reward', total_reward, episode)
        summary_writer.add_scalar('Average Loss', avg_loss, episode)
        summary_writer.add_scalar('Epsilon', agent.epsilon, episode)
        summary_writer.add_scalar('Time Elapsed', elapsed_time, episode)
        summary_writer.add_scalar('Level', info.get('level', 1), episode)
        summary_writer.add_scalar('Lines Cleared', info.get('lines', 0), episode)
        summary_writer.add_scalar('Steps Taken', step_count, episode)
        summary_writer.flush()

        # 打印日志信息
        logger.info(f"Episode {episode}/{num_episodes} | "
              f"得分: {total_reward} | "
              f"等级: {info.get('level', 1)} | "
              f"行数: {info.get('lines', 0)} | "
              f"步数: {step_count} | "
              f"ε: {agent.epsilon:.3f} | "
              f"损失: {avg_loss:.4f} | "
              f"耗时: {elapsed_time:.1f}秒")
        
        # 每50轮保存一次模型
        if episode % 50 == 0:
            ensure_dir("models/checkpoints")
            agent.save_model(f"models/checkpoints/model_ep{episode}.pth")
    
    return scores, all_losses


def test_agent(env: TetrisEnv, agent: DQNAgent):
    """
    测试智能体
    
    参数:
        env: 游戏环境
        agent: DQN智能体
    """
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    
    logger.info(f"测试结束! 总得分: {total_reward}")
    logger.info(f"消除行数: {info.get('lines', 0)}")
    logger.info(f"最终等级: {info.get('level', 1)}")


def main():
    """主函数：训练或测试DQN智能体"""
    # 创建环境
    env = TetrisEnv(render_mode='human')
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # 初始化智能体
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        lr=LEARNING_RATE,
        batch_size=64,
        buffer_capacity=REPLAY_BUFFER_CAPACITY,
        target_update=1000
    )
    logger.info("智能体已初始化")
    
    # 训练模式或测试模式
    train_mode = True
    ensure_dir("models")
    
    if train_mode:
        logger.info("开始训练智能体...")
        logger.info(f"状态空间: {state_shape}, 动作空间: {n_actions}")
        logger.info(f"缓冲区容量: {REPLAY_BUFFER_CAPACITY}")
        logger.info(f"批量大小: {agent.batch_size}")
        logger.info(f"学习率: {LEARNING_RATE}")
        
        # 开始训练
        train_agent(env, agent, NUM_EPISODES)
        
        # 保存最终模型
        agent.save_model("models/final_model.pth")
        
        logger.info("训练完成!")
    else:
        # 加载预训练模型
        model_path = "models/best_model.pth"
        if os.path.exists(model_path):
            logger.info(f"加载模型: {model_path}")
            agent.load_model(model_path)
        else:
            logger.info(f"警告: 未找到模型文件 {model_path}, 将使用随机权重")
        
        # 设置探索率为最小值
        agent.epsilon = agent.epsilon_min
        
        logger.info("开始测试智能体...")
        test_agent(env, agent)
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
        pygame.quit()
    except Exception as e:
        logger.info(f"发生错误: {e}")
        pygame.quit()