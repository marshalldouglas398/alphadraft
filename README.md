# AlphaDraft

AlphaDraft is an experimental fantasy football draft simulator and self-play trainer. It combines historical player projection modeling with an AlphaZero-style policy/value network that learns draft strategy through simulated snake drafts.

The project currently targets an 8-team, 15-round, half-PPR fantasy football format and evaluates against the 2024 season.

## What It Does

- Builds historical player features from `fantasy_data.csv`
- Trains projection and calibration models for player value
- Simulates fantasy drafts with roster constraints and opponent behavior
- Trains a PyTorch policy/value model through self-play
- Uses MCTS-guided draft decisions during training
- Produces draft boards, team results, validation metrics, training history, and model checkpoints

## Project Layout

```text
.
+-- fantasy_alphadraft_zero.py   # Main AlphaZero-style self-play trainer
+-- fantasy_simulation.py        # Feature engineering, projections, baseline simulator
+-- fantasy_data.csv             # Historical fantasy football input data
+-- outputs_alphadraft_zero/     # Main training outputs
+-- outputs_smoke_test/          # Small smoke-test run outputs
+-- .gitignore
`-- LICENSE
```

## Requirements

Python 3.10+ is recommended.

Core dependencies:

```text
numpy
pandas
scikit-learn
torch
```

Install them with:

```bash
pip install numpy pandas scikit-learn torch
```

For CUDA training, install the PyTorch build that matches your GPU and driver from the official PyTorch installation selector.

## Data

The default input file is:

```text
fantasy_data.csv
```

The scripts expect historical fantasy football data with columns such as player name, team, position, age, games, season, experience, passing attempts, carries, targets, and half-PPR fantasy points. `fantasy_simulation.py` normalizes the raw column names into the internal schema used by both scripts.

## Running AlphaDraft

Run the main self-play trainer:

```bash
python fantasy_alphadraft_zero.py
```

By default this will:

- Load `fantasy_data.csv`
- Train on historical seasons beginning in 2010
- Validate on 2022 and 2023
- Run a deterministic 2024 test draft
- Write outputs to `outputs_alphadraft_zero/`

### Faster Smoke Test

For a shorter run:

```bash
python fantasy_alphadraft_zero.py \
  --iterations 1 \
  --episodes-per-iteration 4 \
  --mcts-simulations 2 \
  --shortlist-size 12 \
  --validation-team-count 2 \
  --output-dir outputs_smoke_test
```

### Useful Options

```bash
python fantasy_alphadraft_zero.py \
  --csv-path fantasy_data.csv \
  --output-dir outputs_alphadraft_zero \
  --iterations 10 \
  --episodes-per-iteration 48 \
  --mcts-simulations 40 \
  --shortlist-size 48 \
  --device auto
```

Common flags:

- `--device`: `auto`, `cpu`, `cuda`, or a specific torch device
- `--checkpoint`: warm-start from a saved checkpoint
- `--iterations`: number of self-play training iterations
- `--episodes-per-iteration`: self-play episodes per iteration
- `--mcts-simulations`: MCTS simulations per decision
- `--mcts-simulations-final`: optional ramp target for later iterations
- `--shortlist-size`: candidate pool size per draft pick
- `--shortlist-size-final`: optional ramp target for later iterations
- `--validation-team-count`: number of teams to validate per season, with `0` meaning all teams
- `--training-league-sizes`: optional league-size augmentation values
- `--ensemble-checkpoints`: checkpoint paths to include in ensemble inference

## Outputs

The AlphaDraft trainer writes:

```text
outputs_alphadraft_zero/
+-- alphadraft_zero_model.pt
+-- run_metadata.json
+-- training_history.csv
+-- validation_results.csv
+-- test_2024_draft_board.csv
`-- test_2024_team_results.csv
```

Notes:

- `alphadraft_zero_model.pt` is ignored by git because trained model artifacts can be large.
- `run_metadata.json` records the configuration and feature set used for the run.
- `test_2024_draft_board.csv` contains the final simulated draft board.
- `test_2024_team_results.csv` summarizes simulated team performance.

## Baseline Simulator

The baseline projection and draft simulator can be run directly:

```bash
python fantasy_simulation.py
```

It writes reports to `outputs/`, including model comparisons, 2024 projections, draft board, team rosters, weekly lineups, and summary reports. This script also provides the feature engineering and simulation helpers used by the AlphaDraft trainer.

## Reproducibility

The project uses a fixed random seed by default:

```text
RANDOM_SEED = 42
```

Training is still hardware-sensitive when running with CUDA because PyTorch and GPU kernels can introduce small numerical differences.

## Current Status

AlphaDraft is research code, not a packaged library. The main workflow is script-based, and the output CSVs are intended for inspection, comparison, and iteration on draft strategy.
