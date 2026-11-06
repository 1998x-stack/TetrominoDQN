# Portfolio Engineering Improvement History

### 2026-11-05 15:20 UTC — ChatGPT Portfolio Iteration
- Run ID: 20261105T152001Z-iteration11
- Repo / Base SHA: 1998x-stack/TetrominoDQN / 1e6d63bfa0cc088034408e91f3d8330e02a0962f
- Historical context: First initialization; default-branch docs/improvements.md returned GitHub 404. No existing PRs found by repository PR search; only three earlier commits returned.
- Product problem & expected user value: Training observations may be mutated after insertion, corrupting the historical experiences used by the DQN learner. Snapshotting preserves what the agent actually observed.
- Engineering diagnosis & evidence: replay_buffer.py ReplayBuffer.push stores caller-owned ndarray references; subsequent in-place changes alter sampled experiences. ReplayBuffer accepts zero capacity and zero batch size without explicit error.
- Options & decision: Snapshot aliasing (benefit 5 + correctness 5 + evidence 5 + testability 5 - cost 1 - risk 1 = 18); reject invalid buffer sizes (3+3+5+5-1-1=14); investigate DQN loss batch-one broadcasting (4+4+4+3-3-3=9). Choose snapshotting with closely related size guards; no algorithm or model schema changes.
- Changes: replay_buffer.py copies state and next_state at insertion and validates positive capacity and batch size; tests/test_replay_buffer.py covers mutation isolation, eviction, insufficient batch and invalid sizes. Public sample tuple format is unchanged.
- Validation: B (baseline, local behaviorally equivalent replay-buffer reproduction): python -m unittest discover -s tests -v => 2 failures / 3 tests (mutation isolation and zero capacity). A (patched local reproduction): same command => 3 tests passed. C: python -m compileall -q replay_buffer.py tests => passed. D: exact GitHub branch checkout and full project tests not executed; connector code could not be mounted into local runner. CI not claimed.
- Security & regression review: No credentials, workflows, production writes, or model schema changes; array copies incur memory/time overhead proportional to observation size. Existing eviction and sampling semantics checked locally.
- Risk & rollback: Memory overhead of copied observations; revert branch commits or close the Draft PR, leaving main unchanged.
- Follow-up: Verify tests against a fresh checkout with repository-pinned numpy; inspect DQN loss behavior with batch size one.
- Commit: 4f3a1bbff2d449b8355b8baa84979dc206e368c7 (initial history commit; later metadata-only update follows)
- PR: https://github.com/1998x-stack/TetrominoDQN/pull/1
