# Text2Motion Project

### Data Collection
The `data_collection.py` script is used to collect episodes of humanoid motion data using the Cross-Entropy Method (CEM) for planning. Here's how to run it:

```bash
python examples/data_collection.py --sequence simple_stand --num_episodes 1000 --steps_per_episode 300 --replan_every 1 --torque_std 0.1 --output_dir ./dataset --seed 42
```

**Arguments:**
- `--sequence`: Reference sequence for the task (e.g., `simple_stand` or `balance`).
- `--num_episodes`: Number of episodes to collect.
- `--steps_per_episode`: Number of steps per episode.
- `--replan_every`: Frequency of replanning with CEM.
- `--torque_std`: Standard deviation of torque noise.
- `--output_dir`: Directory to save the collected data.
- `--seed`: Random seed for reproducibility.

### Replay Episode
The `replay_episode.py` script allows you to replay recorded episodes in the MuJoCo UI. You can choose to replay either the control sequence or the state sequence.

**Replay Control Sequence:**
```bash
python replay_episode.py --episode ./dataset/episode_0000.npz --sequence simple_stand --frequency 30 --show_reference
```

**Replay State Sequence:**
```bash
python replay_episode.py --episode ./dataset/episode_0000.npz --sequence simple_stand --frequency 30 --show_reference --replay_mode state
```

**Arguments:**
- `--episode`: Path to the episode `.npz` file.
- `--sequence`: Reference sequence for visualization.
- `--frequency`: Playback frequency in Hz.
- `--show_reference`: Show reference trajectory as a transparent ghost.
- `--replay_mode`: Mode of replay, either `control` (default) or `state`.

