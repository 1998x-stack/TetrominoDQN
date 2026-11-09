"""Headless deterministic regression for actual Tetris mechanics (not DQN accuracy)."""
import random
import unittest

import numpy as np
from config import GRID_HEIGHT, GRID_WIDTH
from tetris_env import TetrisEnv
from tetromino import Tetromino


class TetrisMechanicsTests(unittest.TestCase):
    def setUp(self):
        self.env = TetrisEnv()
        self.env.reset(seed=2026)

    def tearDown(self):
        self.env.close()

    def test_sideways_and_down_moves_mutate_piece_position(self):
        x, y = self.env.current_piece.x, self.env.current_piece.y
        obs, reward, done, info = self.env.step(0)
        self.assertFalse(done)
        self.assertEqual(self.env.current_piece.x, x - 1)
        self.assertEqual(self.env.current_piece.y, y)
        self.assertEqual(reward, 0.1)
        self.env.step(1)
        self.assertEqual(self.env.current_piece.x, x)
        self.env.step(2)
        self.assertEqual(self.env.current_piece.y, y + 1)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_wall_rejects_move_and_keeps_coordinates(self):
        self.env.current_piece = Tetromino(4)
        self.env.current_piece.x = 0
        self.assertFalse(self.env._move(-1, 0))
        self.assertEqual(self.env.current_piece.x, 0)
        self.assertEqual(self.env.step(0)[1], -0.1)

    def test_rotation_checks_wall_and_does_not_mutate_on_failure(self):
        piece = Tetromino(0)
        piece.shape = [[1], [1], [1], [1]]
        piece.x = GRID_WIDTH - 1
        self.env.current_piece = piece
        self.assertFalse(self.env._rotate_piece())
        self.assertEqual(piece.shape, [[1], [1], [1], [1]])
        piece.x = 3
        self.assertTrue(self.env._rotate_piece())
        self.assertEqual(piece.shape, [[1, 1, 1, 1]])

    def test_two_completed_rows_are_removed_together(self):
        self.env.board[-1] = [1] * GRID_WIDTH
        self.env.board[-2] = [1] * GRID_WIDTH
        self.env._clear_lines()
        self.assertEqual(self.env.lines_cleared, 2)
        self.assertEqual(self.env.score, 300)
        self.assertEqual(len(self.env.board), GRID_HEIGHT)
        self.assertTrue(all(cell == 0 for row in self.env.board for cell in row))

    def test_hard_drop_locks_piece_and_does_not_move_next_piece(self):
        self.env.current_piece = Tetromino(4)
        self.env.next_piece = Tetromino(4)
        obs, reward, done, info = self.env.step(4)
        self.assertFalse(done)
        self.assertAlmostEqual(reward, 0.5)
        self.assertEqual(self.env.current_piece.y, 0)
        self.assertEqual(self.env.current_piece.shape_index, 4)
        self.assertEqual(sum(bool(v) for row in self.env.board for v in row), 4)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_seed_controls_pieces_but_does_not_touch_global_rng(self):
        previous = random.getstate()
        self.env.reset(seed=99)
        first = (self.env.current_piece.shape_index, self.env.next_piece.shape_index)
        self.env.reset(seed=99)
        second = (self.env.current_piece.shape_index, self.env.next_piece.shape_index)
        self.assertEqual(first, second)
        self.assertEqual(previous, random.getstate())

    def test_episode_truncation_freezes_game_until_reset(self):
        self.env.max_steps = 1
        obs, reward, done, info = self.env.step(5)
        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        snapshot = [row[:] for row in self.env.board]
        with self.assertRaises(RuntimeError):
            self.env.step(1)
        self.assertEqual(self.env.board, snapshot)
        self.env.reset(seed=99)
        self.assertFalse(self.env._episode_done)

    def test_invalid_action_does_not_advance_game(self):
        with self.assertRaises(ValueError):
            self.env.step(99)
        self.assertEqual(self.env.steps, 0)

    def test_short_real_episode_has_valid_observations(self):
        rng = random.Random(123)
        self.env.reset(seed=42)
        done = False
        for _ in range(60):
            obs, reward, done, info = self.env.step(rng.randrange(6))
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            if done:
                break
        self.assertGreater(self.env.steps, 0)


if __name__ == "__main__":
    unittest.main()
