# TetrominoDQN · Tetris 规则环境与 DQN 研究工程

本仓库包含 Pygame 俄罗斯方块环境、遗留 Gym 四元组 API、经验回放、DQN 模型及训练入口。**规则回归、随机策略基线、深度网络训练是三个不同的验证等级**，不要将轻量 CPU 测试当作训练成功或实际游戏品质认证。

## P2 玩法核心修复（2026-11-09）

- 修复 `_move` 只判断而**未更新方块坐标**，导致左移、右移、下移和自动重力不真正移动的问题。
- 同一时刻消除两行以上时，改为一次性移除所有满行，再补足顶部空行，避免边删除边插入造成索引错位。
- 硬降落后不会在同一步继续推动刚生成的下一块；墙边不能旋转时不会篡改原形状。
- `reset(seed=...)` 使用每个环境自己的随机源，可独立复现方块序列，不污染 Python 的全局随机状态。
- 非法动作明确拒绝；步数截断/堆叠死亡后禁止继续 `step`，需要重置游戏。
- 新增无界面玩法回归与种子固定的 **uniform-random-actions** 基线（真实执行对局，不启动 DQN 神经网络）。

### 复现最小环境（无需 GPU）

Python 3.11 推荐从**本仓库根目录**运行：

```bash
git clone https://github.com/1998x-stack/TetrominoDQN.git
cd TetrominoDQN
python -m pip install numpy==1.26.4 pygame==2.5.2 gym==0.26.2
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m unittest discover -s tests -v
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python scripts/random_baseline.py --episodes 3 --steps 400
```

`tetris_env.TetrisEnv` 暂沿用旧训练入口要求的 **`reset()->obs` 和 `step()->(obs,reward,done,info)`**，与现代 Gymnasium 五元组 API 不等价。此阶段优先保障原训练调用路径与实际规则一致，不声称完成全量现代 API 升级。

本分支新增 CI `P2 Tetris gameplay contract`：检查 Python 语法、移动/旋转/硬降落、两行消除、越界与重置、观察类型、旧 replay buffer，以及 3×3 组固定种子的随机策略对局 JSON **双重一致**。原始 CI 应以实际运行结果判断是否通过，链接：[工作流](https://github.com/1998x-stack/TetrominoDQN/actions/workflows/tetris-gameplay.yml)。

## 训练与项目结构

- `tetris_env.py` + `tetromino.py`：方块规则、状态空间与可选 Pygame 渲染。
- `replay_buffer.py` + `dqn.py` + `dqn_agent.py`：经验回放与深度 Q 学习组件。
- `train.py`：历史默认训练/演示入口，依赖 PyTorch、TensorBoardX 和显示设备；请参考 `requirements.txt`，并核对对应版本与训练输出。
- `tests/test_tetris_mechanics.py`：规则、终局、安全性和随机种子回归。
- `scripts/random_baseline.py`：固定三个种子、固定回合与步数的无学习随机动作基准，输出分数、行数与存活步数。

**尚未验收：** DQN 完整训练奖励曲线、最终模型复现、随机/智能体同条件对照与真实物理键盘/屏幕体验。原 README 曾列出“平均分 5000+、100+ 消行”等示例数据，但未附足够可独立核对的训练证据，因此不将其视为已验证性能。

## 授权与素材

提交和使用代码前请检查本仓库具体许可证文件及 Pygame/其他依赖的许可证。此阶段未增加第三方 ROM、角色美术或训练权重，亦不声明未经验证的再分发授权。
