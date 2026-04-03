# Locomotion Replay

Replay recorded G1 locomotion episodes from NPZ datasets using Genesis.

## Prerequisites

- Genesis installed (`pip install genesis`)
- Python 3.10+, PyTorch, NumPy, tqdm

## Files

```
locomotion/
├── replay.py              # Replay script
├── assets/
│   └── g1_29dof_rev_1_0.xml  # G1 robot MJCF model
├── dataset1.npz           # Recorded episodes
├── dataset2.npz
├── dataset3.npz
├── dataset4.npz
└── dataset5.npz
```

## Usage

### View in interactive viewer

```bash
python replay.py -f dataset1.npz --vis
```

### Record to video (camera follows robot)

```bash
python replay.py -f dataset1.npz --rec -o replay1.mp4
```

### Specify a different episode

Each NPZ file contains episodes named `demo_0`, `demo_1`, etc. Default is `demo_0`.

```bash
python replay.py -f dataset1.npz --rec -e demo_1 -o replay1_ep1.mp4
```

### Use CPU backend

```bash
python replay.py -f dataset1.npz --vis --cpu
```

### Record all datasets

```bash
for i in 1 2 3 4 5; do
    python replay.py -f dataset${i}.npz --rec -o replay${i}.mp4
done
```

## Options

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--file` | `-f` | Path to NPZ dataset file | (required) |
| `--episode` | `-e` | Episode name to replay | `demo_0` |
| `--vis` | `-v` | Show interactive viewer | off |
| `--rec` | `-r` | Record video | off |
| `--output` | `-o` | Output video filename | `replay.mp4` |
| `--cpu` | | Use CPU backend | GPU |
