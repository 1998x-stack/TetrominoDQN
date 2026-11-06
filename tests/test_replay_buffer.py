"""Regression tests for replay observation ownership and boundary behavior."""
import unittest
import numpy as np
from replay_buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_snapshot_prevents_mutation(self):
        buffer = ReplayBuffer(2)
        state = np.array([1., 2.])
        next_state = np.array([3., 4.])
        buffer.push(state, 0, 1., next_state, False)
        state[0] = 99
        next_state[0] = 88
        sample = buffer.sample(1)
        self.assertEqual(sample[0][0, 0], 1.)
        self.assertEqual(sample[3][0, 0], 3.)

    def test_batch_and_eviction(self):
        buffer = ReplayBuffer(2)
        self.assertIsNone(buffer.sample(1))
        for i in range(3):
            buffer.push(np.array([i]), i, float(i), np.array([i + 1]), False)
        self.assertTrue(buffer.is_full())
        self.assertEqual(set(buffer.sample(2)[1].tolist()), {1, 2})

    def test_invalid_capacity_and_batch(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(0)
        buffer = ReplayBuffer(2)
        with self.assertRaises(ValueError):
            buffer.sample(0)


if __name__ == "__main__":
    unittest.main()
