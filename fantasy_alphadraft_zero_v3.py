from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from fantasy_simulation import (
    BENCH_SOFT_CAPS,
    DRAFT_ROUNDS,
    FLEX_POSITIONS,
    FLEX_SLOTS,
    LEAGUE_SIZE,
    PRIMARY_STARTERS,
    RANDOM_SEED,
    build_features,
    build_pick_slots,
    get_model_features,
    load_data,
    simulate_team_weekly_lineups,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - runtime guard only
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


@dataclass
class AlphaDraftConfig:
    csv_path: Path = Path("fantasy_data.csv")
    output_dir: Path = Path("outputs_alphadraft_zero")
    min_train_season: int = 2010
    validation_seasons: Tuple[int, int] = (2022, 2023)
    test_season: int = 2024
    league_size: int = LEAGUE_SIZE
    draft_rounds: int = DRAFT_ROUNDS
    max_players_per_season: int = 220
    shortlist_size: int = 48
    mcts_simulations: int = 40
    mcts_focal_depth: int = 2
    self_play_iterations: int = 10
    episodes_per_iteration: int = 48
    training_epochs_per_iteration: int = 2
    replay_buffer_size: int = 6000
    batch_size: int = 32
    hidden_dim: int = 256
    dropout: float = 0.15
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    entropy_bonus_weight: float = 0.01
    label_smoothing: float = 0.02
    dirichlet_alpha: float = 0.30
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.30
    opponent_temperature: float = 0.85
    early_round_temperature: float = 1.00
    late_round_temperature: float = 0.35
    late_round_start: int = 8
    validation_patience: int = 4
    log_every_episodes: int = 1
    seed: int = RANDOM_SEED
    device: str = "auto"


@dataclass
class SeasonPool:
    season: int
    players: pd.DataFrame
    static_features: np.ndarray
    public_score: np.ndarray
    public_score_z: np.ndarray


@dataclass
class TrainingExample:
    season: int
    context: np.ndarray
    candidates: np.ndarray
    policy: np.ndarray
    reward: float


@dataclass
class EpisodeSummary:
    season: int
    focal_team_index: int
    focal_points: float
    focal_reward: float
    focal_rank: int
    winner_points: float
    examples: List[TrainingExample]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[TrainingExample] = deque(maxlen=capacity)

    def extend(self, examples: Sequence[TrainingExample]) -> None:
        self.buffer.extend(examples)

    def __len__(self) -> int:
        return len(self.buffer)

    def as_list(self) -> List[TrainingExample]:
        return list(self.buffer)


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class FeatureEncoder:
    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_medians: pd.Series | None = None
        self.numeric_means: pd.Series | None = None
        self.numeric_stds: pd.Series | None = None
        self.category_levels: Dict[str, List[str]] = {}
        self.feature_names: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        numeric_frame = train_df[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        self.numeric_medians = numeric_frame.median()
        numeric_filled = numeric_frame.fillna(self.numeric_medians)
        self.numeric_means = numeric_filled.mean()
        self.numeric_stds = numeric_filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

        self.category_levels = {}
        for column in self.categorical_features:
            values = train_df[column].fillna("__UNK__").astype(str)
            levels = sorted(set(values.tolist()) | {"__UNK__"})
            self.category_levels[column] = levels

        self.feature_names = list(self.numeric_features)
        for column in self.categorical_features:
            self.feature_names.extend([f"{column}={level}" for level in self.category_levels[column]])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.numeric_medians is None or self.numeric_means is None or self.numeric_stds is None:
            raise RuntimeError("FeatureEncoder.fit must be called before transform.")

        numeric_frame = (
            df[self.numeric_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(self.numeric_medians)
        )
        numeric_scaled = ((numeric_frame - self.numeric_means) / self.numeric_stds).to_numpy(dtype=np.float32)

        categorical_blocks: List[np.ndarray] = []
        for column in self.categorical_features:
            levels = self.category_levels[column]
            level_to_idx = {level: idx for idx, level in enumerate(levels)}
            values = df[column].fillna("__UNK__").astype(str)
            mapped = values.where(values.isin(level_to_idx), "__UNK__")
            block = np.zeros((len(df), len(levels)), dtype=np.float32)
            for row_idx, level in enumerate(mapped):
                block[row_idx, level_to_idx[level]] = 1.0
            categorical_blocks.append(block)

        if categorical_blocks:
            return np.concatenate([numeric_scaled, *categorical_blocks], axis=1).astype(np.float32)
        return numeric_scaled.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyValueNet(nn.Module):
    def __init__(self, context_dim: int, candidate_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.context_mlp = MLP(context_dim, hidden_dim, hidden_dim, dropout)
        self.candidate_mlp = MLP(candidate_dim, hidden_dim, hidden_dim, dropout)
        self.interaction_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.logit_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, candidate_count, _ = candidates.shape
        context_hidden = self.context_mlp(context)
        candidate_hidden = self.candidate_mlp(candidates.reshape(batch_size * candidate_count, -1))
        candidate_hidden = candidate_hidden.reshape(batch_size, candidate_count, -1)
        context_expanded = context_hidden.unsqueeze(1).expand(-1, candidate_count, -1)
        pair_hidden = self.interaction_mlp(torch.cat([context_expanded, candidate_hidden], dim=-1))

        logits = self.logit_head(pair_hidden).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)

        mask_float = mask.float().unsqueeze(-1)
        pooled_mean = (pair_hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        masked_pair = pair_hidden.masked_fill(~mask.unsqueeze(-1), -1e9)
        pooled_max = masked_pair.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        value = self.value_head(torch.cat([context_hidden, pooled_mean, pooled_max], dim=-1)).squeeze(-1)
        return logits, value


@dataclass
class SearchNode:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    candidate_indices: np.ndarray | None = None
    children: Dict[int, "SearchNode"] | None = None

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class DraftEnvironment:
    def __init__(self, season_pool: SeasonPool, config: AlphaDraftConfig):
        self.pool = season_pool
        self.config = config
        self.pick_slots = build_pick_slots(config.league_size, config.draft_rounds)
        self.available_mask = np.ones(len(self.pool.players), dtype=bool)
        self.rosters: List[List[int]] = [[] for _ in range(config.league_size)]
        self.slot_idx = 0
        self.draft_history: List[dict] = []

    def clone(self) -> "DraftEnvironment":
        clone = DraftEnvironment.__new__(DraftEnvironment)
        clone.pool = self.pool
        clone.config = self.config
        clone.pick_slots = self.pick_slots
        clone.available_mask = self.available_mask.copy()
        clone.rosters = [list(roster) for roster in self.rosters]
        clone.slot_idx = self.slot_idx
        clone.draft_history = list(self.draft_history)
        return clone

    def is_terminal(self) -> bool:
        return self.slot_idx >= len(self.pick_slots)

    def current_slot(self) -> dict:
        return self.pick_slots[self.slot_idx]

    def current_team(self) -> int:
        return int(self.current_slot()["team_index"])

    def current_round(self) -> int:
        return int(self.current_slot()["round"])

    def roster_counts(self, team_index: int) -> Dict[str, int]:
        counts = {position: 0 for position in PRIMARY_STARTERS}
        for player_idx in self.rosters[team_index]:
            position = str(self.pool.players.iloc[player_idx]["position"])
            counts[position] += 1
        return counts

    def flex_filled(self, counts: Dict[str, int]) -> int:
        excess = (
            max(counts["RB"] - PRIMARY_STARTERS["RB"], 0)
            + max(counts["WR"] - PRIMARY_STARTERS["WR"], 0)
            + max(counts["TE"] - PRIMARY_STARTERS["TE"], 0)
        )
        return min(excess, FLEX_SLOTS)

    def starter_gap(self, team_index: int, position: str) -> float:
        counts = self.roster_counts(team_index)
        base_gap = max(PRIMARY_STARTERS[position] - counts[position], 0)
        if position in FLEX_POSITIONS:
            flex_gap = max(FLEX_SLOTS - self.flex_filled(counts), 0)
            return float(base_gap + (0.5 * flex_gap))
        return float(base_gap)

    def legal_actions(self, team_index: int) -> np.ndarray:
        available_indices = np.flatnonzero(self.available_mask)
        if available_indices.size == 0:
            return available_indices

        counts = self.roster_counts(team_index)
        eligible: List[int] = []
        for idx in available_indices:
            position = str(self.pool.players.iloc[idx]["position"])
            if counts[position] >= BENCH_SOFT_CAPS[position]:
                continue
            if position == "QB" and counts["QB"] >= 2:
                continue
            if position == "TE" and counts["TE"] >= 2:
                continue
            eligible.append(int(idx))
        if eligible:
            return np.asarray(eligible, dtype=int)
        return available_indices.astype(int)

    def remaining_counts_by_position(self) -> Dict[str, int]:
        available_df = self.pool.players.loc[self.available_mask]
        return {
            position: int(available_df["position"].eq(position).sum())
            for position in PRIMARY_STARTERS
        }

    def remaining_starter_demand(self, position: str) -> int:
        demand = 0
        for team_index in range(self.config.league_size):
            counts = self.roster_counts(team_index)
            demand += max(PRIMARY_STARTERS[position] - counts[position], 0)
        if position in FLEX_POSITIONS:
            open_flex = 0
            for team_index in range(self.config.league_size):
                counts = self.roster_counts(team_index)
                open_flex += max(FLEX_SLOTS - self.flex_filled(counts), 0)
            demand += int(math.ceil(open_flex / len(FLEX_POSITIONS)))
        return max(demand, 1)

    def position_need_multiplier(self, team_index: int, position: str) -> float:
        counts = self.roster_counts(team_index)
        starter_need = max(PRIMARY_STARTERS[position] - counts[position], 0)
        flex_need = max(FLEX_SLOTS - self.flex_filled(counts), 0) if position in FLEX_POSITIONS else 0
        round_number = self.current_round()

        if position == "QB":
            if counts["QB"] == 0:
                return min(1.12, 0.66 + (0.08 * round_number))
            if counts["QB"] == 1:
                return min(0.52, 0.10 + (0.06 * max(round_number - 8, 0)))
            return 0.05

        if position == "TE":
            if counts["TE"] == 0:
                return min(1.08, 0.72 + (0.06 * round_number))
            if counts["TE"] == 1:
                return min(0.58, 0.14 + (0.05 * max(round_number - 8, 0)))
            return 0.08

        if starter_need > 0:
            if position == "RB":
                multiplier = 1.24 + (0.12 * starter_need)
            elif position == "WR":
                multiplier = 1.12 + (0.08 * starter_need)
            else:
                multiplier = 1.18 + (0.10 * starter_need)
        elif flex_need > 0:
            multiplier = 1.10 if position == "RB" else 1.00
        else:
            multiplier = 0.92 if position == "RB" else 0.84

        if counts[position] >= BENCH_SOFT_CAPS[position]:
            multiplier *= 0.60
        return float(multiplier)

    def shortlist_score(self, team_index: int, player_idx: int) -> float:
        player = self.pool.players.iloc[player_idx]
        position = str(player["position"])
        remaining = self.remaining_counts_by_position()
        need = self.position_need_multiplier(team_index, position)
        scarcity = self.remaining_starter_demand(position) / max(remaining[position], 1)
        counts = self.roster_counts(team_index)
        stack = self.stack_signal(team_index, player_idx)
        duplicate_penalty = self.same_team_penalty(team_index, player_idx)
        qb_penalty = 0.25 if position == "QB" and counts["QB"] >= 1 and self.current_round() <= 10 else 0.0
        return float(
            self.pool.public_score_z[player_idx]
            + (0.55 * np.log1p(max(need, 0.0)))
            + (0.45 * scarcity)
            + (0.12 * stack)
            - (0.14 * duplicate_penalty)
            - qb_penalty
        )

    def shortlist(self, team_index: int, limit: int) -> np.ndarray:
        legal = self.legal_actions(team_index)
        if legal.size <= limit:
            return legal
        scores = np.asarray([self.shortlist_score(team_index, idx) for idx in legal], dtype=float)
        top_order = np.argsort(scores)[::-1][:limit]
        return legal[top_order]

    def stack_signal(self, team_index: int, player_idx: int) -> float:
        player = self.pool.players.iloc[player_idx]
        team = str(player["team"])
        position = str(player["position"])
        roster_df = self.pool.players.iloc[self.rosters[team_index]] if self.rosters[team_index] else pd.DataFrame()
        if roster_df.empty:
            return 0.0
        if position == "QB":
            return float(((roster_df["team"] == team) & (roster_df["position"].isin(["WR", "TE"]))).sum())
        if position in {"WR", "TE"}:
            return float(((roster_df["team"] == team) & (roster_df["position"] == "QB")).sum())
        return 0.0

    def same_team_penalty(self, team_index: int, player_idx: int) -> float:
        player = self.pool.players.iloc[player_idx]
        team = str(player["team"])
        roster_df = self.pool.players.iloc[self.rosters[team_index]] if self.rosters[team_index] else pd.DataFrame()
        if roster_df.empty:
            return 0.0
        return float((roster_df["team"] == team).sum())

    def build_context_vector(self, team_index: int) -> np.ndarray:
        round_norm = self.current_round() / max(self.config.draft_rounds, 1)
        pick_norm = (self.slot_idx + 1) / max(len(self.pick_slots), 1)
        available_frac = float(self.available_mask.mean())
        counts = self.roster_counts(team_index)
        flex_gap = max(FLEX_SLOTS - self.flex_filled(counts), 0)

        other_counts = np.array(
            [
                list(self.roster_counts(other_idx).values())
                for other_idx in range(self.config.league_size)
                if other_idx != team_index
            ],
            dtype=float,
        )
        if other_counts.size == 0:
            other_counts = np.zeros((1, 4), dtype=float)

        remaining_df = self.pool.players.loc[self.available_mask]
        remaining_top = []
        remaining_frac = []
        total_remaining = max(len(remaining_df), 1)
        for position in ["QB", "RB", "WR", "TE"]:
            pos_df = remaining_df[remaining_df["position"] == position]
            remaining_top.append(float(pos_df["public_score"].max()) if not pos_df.empty else 0.0)
            remaining_frac.append(float(len(pos_df) / total_remaining))

        context = np.array(
            [
                round_norm,
                pick_norm,
                available_frac,
                counts["QB"] / BENCH_SOFT_CAPS["QB"],
                counts["RB"] / BENCH_SOFT_CAPS["RB"],
                counts["WR"] / BENCH_SOFT_CAPS["WR"],
                counts["TE"] / BENCH_SOFT_CAPS["TE"],
                self.starter_gap(team_index, "QB"),
                self.starter_gap(team_index, "RB"),
                self.starter_gap(team_index, "WR"),
                self.starter_gap(team_index, "TE"),
                float(flex_gap),
                other_counts[:, 0].mean() / BENCH_SOFT_CAPS["QB"],
                other_counts[:, 1].mean() / BENCH_SOFT_CAPS["RB"],
                other_counts[:, 2].mean() / BENCH_SOFT_CAPS["WR"],
                other_counts[:, 3].mean() / BENCH_SOFT_CAPS["TE"],
                *remaining_top,
                *remaining_frac,
            ],
            dtype=np.float32,
        )
        return context

    def build_candidate_matrix(self, team_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        counts = self.roster_counts(team_index)
        remaining = self.remaining_counts_by_position()
        dynamic_rows: List[List[float]] = []
        for idx in candidate_indices:
            player = self.pool.players.iloc[int(idx)]
            position = str(player["position"])
            starter_gap = self.starter_gap(team_index, position)
            scarcity = self.remaining_starter_demand(position) / max(remaining[position], 1)
            dynamic_rows.append(
                [
                    self.position_need_multiplier(team_index, position),
                    counts[position] / BENCH_SOFT_CAPS[position],
                    starter_gap,
                    1.0 if position in FLEX_POSITIONS else 0.0,
                    self.stack_signal(team_index, int(idx)),
                    self.same_team_penalty(team_index, int(idx)),
                    scarcity,
                    float(self.pool.public_score_z[int(idx)]),
                    self.current_round() / max(self.config.draft_rounds, 1),
                ]
            )
        dynamic_matrix = np.asarray(dynamic_rows, dtype=np.float32)
        return np.concatenate([self.pool.static_features[candidate_indices], dynamic_matrix], axis=1).astype(np.float32)

    def step(self, player_idx: int) -> None:
        slot = self.current_slot()
        team_index = int(slot["team_index"])
        self.rosters[team_index].append(int(player_idx))
        self.available_mask[int(player_idx)] = False

        player = self.pool.players.iloc[int(player_idx)].to_dict()
        player["fantasy_team"] = f"Team {team_index + 1}"
        player["team_index"] = team_index
        player["overall_pick"] = int(slot["overall_pick"])
        player["round"] = int(slot["round"])
        player["pick_in_round"] = int(slot["pick_in_round"])
        self.draft_history.append(player)
        self.slot_idx += 1

    def roster_frame(self, team_index: int) -> pd.DataFrame:
        roster_indices = self.rosters[team_index]
        if not roster_indices:
            return pd.DataFrame(columns=list(self.pool.players.columns))
        roster_df = self.pool.players.iloc[roster_indices].copy().reset_index(drop=True)
        roster_df["fantasy_team"] = f"Team {team_index + 1}"
        roster_df["predicted_points"] = roster_df["public_score"]
        return roster_df

    def evaluate_terminal(self) -> Tuple[pd.DataFrame, np.ndarray]:
        team_rows: List[dict] = []
        for team_index in range(self.config.league_size):
            roster_df = self.roster_frame(team_index)
            total_points, _ = simulate_team_weekly_lineups(
                team_name=f"Team {team_index + 1}",
                roster_df=roster_df,
                point_column="actual_points",
                seed=self.config.seed,
                record_lineups=False,
            )
            team_rows.append(
                {
                    "team_index": team_index,
                    "team_name": f"Team {team_index + 1}",
                    "total_season_points": float(total_points),
                }
            )

        team_results = pd.DataFrame(team_rows).sort_values("total_season_points", ascending=False).reset_index(drop=True)
        team_results["rank"] = np.arange(1, len(team_results) + 1)
        mean_points = float(team_results["total_season_points"].mean())
        std_points = float(team_results["total_season_points"].std(ddof=0))
        std_points = std_points if std_points > 1e-6 else 1.0

        reward_map = np.zeros(self.config.league_size, dtype=np.float32)
        for _, row in team_results.iterrows():
            team_index = int(row["team_index"])
            rank = int(row["rank"])
            rank_reward = 1.0 - (2.0 * (rank - 1) / max(self.config.league_size - 1, 1))
            point_reward = np.clip((float(row["total_season_points"]) - mean_points) / std_points, -2.0, 2.0) / 2.0
            reward_map[team_index] = np.float32((0.70 * rank_reward) + (0.30 * point_reward))
        return team_results, reward_map

    def draft_board_frame(self) -> pd.DataFrame:
        if not self.draft_history:
            return pd.DataFrame()
        return pd.DataFrame(self.draft_history).sort_values("overall_pick").reset_index(drop=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> "torch.device":
    if torch is None:  # pragma: no cover - runtime guard only
        raise SystemExit(
            f"PyTorch is required for fantasy_alphadraft_zero.py. Import failed with: {TORCH_IMPORT_ERROR}"
        )
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_visible_gpu_count(device: "torch.device") -> int:
    if torch is None or device.type != "cuda":
        return 0
    return int(torch.cuda.device_count())


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def build_training_model(
    context_dim: int,
    candidate_dim: int,
    config: AlphaDraftConfig,
    device: "torch.device",
) -> nn.Module:
    model: nn.Module = PolicyValueNet(context_dim, candidate_dim, config.hidden_dim, config.dropout).to(device)
    gpu_count = get_visible_gpu_count(device)
    if gpu_count > 1:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    return model


def build_public_board_score(season_df: pd.DataFrame) -> pd.Series:
    score = (
        0.32 * season_df["historical_best_points_before"].fillna(0.0)
        + 0.22 * season_df["fantasy_points_total_roll3"].fillna(season_df["fantasy_points_total_lag1"]).fillna(0.0)
        + 14.0 * season_df["fantasy_points_per_game_roll3"].fillna(0.0)
        + 0.18 * season_df["historical_best_vor_before"].fillna(0.0)
        + 24.0 * season_df["usage_share_within_team_roll3"].fillna(0.0)
        + 10.0 * season_df["vacated_position_share"].fillna(0.0)
        + 8.0 * season_df["role_growth_signal"].fillna(0.0)
        + 6.0 * season_df["opportunity_shock_score"].fillna(0.0)
        - 8.0 * season_df["prior_year_major_absence"].fillna(0.0)
        - 4.0 * season_df["injury_return_candidate"].fillna(0.0)
        - 1.2 * season_df["age_delta_from_peak"].clip(lower=0.0).fillna(0.0)
    )
    score = score.astype(float)
    position_bonus = season_df["position"].map({"QB": -5.0, "RB": 3.0, "WR": 2.5, "TE": -2.0}).fillna(0.0)
    return score + position_bonus


def build_season_pools(
    feature_df: pd.DataFrame,
    encoder: FeatureEncoder,
    seasons: Sequence[int],
    config: AlphaDraftConfig,
) -> Dict[int, SeasonPool]:
    season_pools: Dict[int, SeasonPool] = {}
    for season in seasons:
        season_df = feature_df[feature_df["season"] == season].copy()
        if season_df.empty:
            continue
        season_df["public_score"] = build_public_board_score(season_df)
        season_df = season_df.sort_values("public_score", ascending=False).head(config.max_players_per_season).copy()
        season_df = season_df.reset_index(drop=True)
        season_df["actual_points"] = season_df["fantasy_points_total"]
        season_df["player_pool_index"] = np.arange(len(season_df))
        public_score = season_df["public_score"].to_numpy(dtype=np.float32)
        public_std = float(public_score.std(ddof=0))
        public_std = public_std if public_std > 1e-6 else 1.0
        public_score_z = ((public_score - public_score.mean()) / public_std).astype(np.float32)
        static_features = encoder.transform(season_df)
        season_pools[season] = SeasonPool(
            season=season,
            players=season_df,
            static_features=static_features,
            public_score=public_score,
            public_score_z=public_score_z,
        )
    return season_pools


def softmax_numpy(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    weights = np.clip(weights, 1e-12, None)
    return weights / weights.sum()


def network_policy_and_value(
    env: DraftEnvironment,
    team_index: int,
    candidate_indices: np.ndarray,
    model: PolicyValueNet,
    device: "torch.device",
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    context = env.build_context_vector(team_index)
    candidate_matrix = env.build_candidate_matrix(team_index, candidate_indices)
    context_tensor = torch.from_numpy(context).to(device).unsqueeze(0)
    candidate_tensor = torch.from_numpy(candidate_matrix).to(device).unsqueeze(0)
    mask_tensor = torch.ones((1, candidate_matrix.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = model(context_tensor, candidate_tensor, mask_tensor)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probabilities.astype(np.float32), float(value.item()), context, candidate_matrix


def blended_policy_prior(
    env: DraftEnvironment,
    team_index: int,
    candidate_indices: np.ndarray,
    model: PolicyValueNet,
    device: "torch.device",
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    network_probs, value, context, candidate_matrix = network_policy_and_value(
        env,
        team_index,
        candidate_indices,
        model,
        device,
    )
    heuristic_scores = np.asarray([env.shortlist_score(team_index, idx) for idx in candidate_indices], dtype=np.float32)
    heuristic_probs = softmax_numpy(heuristic_scores, temperature=0.85)
    blended = (0.72 * network_probs) + (0.28 * heuristic_probs)
    blended = blended / blended.sum()
    return blended.astype(np.float32), value, context, candidate_matrix


def choose_opponent_action(
    env: DraftEnvironment,
    team_index: int,
    model: PolicyValueNet,
    device: "torch.device",
    config: AlphaDraftConfig,
    stochastic: bool,
) -> int:
    candidate_indices = env.shortlist(team_index, config.shortlist_size)
    priors, _, _, _ = blended_policy_prior(env, team_index, candidate_indices, model, device)
    if stochastic:
        probabilities = softmax_numpy(np.log(np.clip(priors, 1e-9, 1.0)), temperature=config.opponent_temperature)
        return int(np.random.choice(candidate_indices, p=probabilities))
    return int(candidate_indices[int(np.argmax(priors))])


def add_dirichlet_noise(priors: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    if priors.size == 0:
        return priors
    noise = np.random.dirichlet(np.full(priors.size, alpha, dtype=np.float32))
    blended = ((1.0 - epsilon) * priors) + (epsilon * noise)
    return (blended / blended.sum()).astype(np.float32)


def expand_node(
    node: SearchNode,
    env: DraftEnvironment,
    perspective_team: int,
    model: PolicyValueNet,
    device: "torch.device",
    config: AlphaDraftConfig,
    add_root_noise: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    candidate_indices = env.shortlist(perspective_team, config.shortlist_size)
    priors, value, context, candidate_matrix = blended_policy_prior(
        env,
        perspective_team,
        candidate_indices,
        model,
        device,
    )
    if add_root_noise:
        priors = add_dirichlet_noise(priors, config.dirichlet_alpha, config.dirichlet_epsilon)
    node.expanded = True
    node.candidate_indices = candidate_indices
    node.children = {
        int(action_idx): SearchNode(prior=float(prior))
        for action_idx, prior in zip(candidate_indices.tolist(), priors.tolist())
    }
    return candidate_indices, priors, context, candidate_matrix, value


def evaluate_leaf_value(
    env: DraftEnvironment,
    perspective_team: int,
    model: PolicyValueNet,
    device: "torch.device",
    config: AlphaDraftConfig,
) -> float:
    candidate_indices = env.shortlist(perspective_team, config.shortlist_size)
    _, value, _, _ = blended_policy_prior(env, perspective_team, candidate_indices, model, device)
    return float(value)


def select_child(node: SearchNode, c_puct: float) -> Tuple[int, SearchNode]:
    assert node.children is not None
    parent_visits = max(node.visit_count, 1)
    best_action = None
    best_score = -np.inf
    best_child = None
    for action, child in node.children.items():
        q_value = child.q_value()
        u_value = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
        score = q_value + u_value
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    if best_action is None or best_child is None:
        raise RuntimeError("select_child called on a node without children.")
    return int(best_action), best_child


def simulate_search(
    node: SearchNode,
    env: DraftEnvironment,
    perspective_team: int,
    model: PolicyValueNet,
    device: "torch.device",
    config: AlphaDraftConfig,
    focal_depth: int,
) -> float:
    if env.is_terminal():
        _, rewards = env.evaluate_terminal()
        return float(rewards[perspective_team])

    current_team = env.current_team()
    if current_team != perspective_team:
        action = choose_opponent_action(env, current_team, model, device, config, stochastic=True)
        env.step(action)
        return simulate_search(node, env, perspective_team, model, device, config, focal_depth)

    if focal_depth >= config.mcts_focal_depth:
        return evaluate_leaf_value(env, perspective_team, model, device, config)

    if not node.expanded:
        _, _, _, _, value = expand_node(node, env, perspective_team, model, device, config, add_root_noise=False)
        return float(value)

    action, child = select_child(node, config.c_puct)
    env.step(action)
    value = simulate_search(child, env, perspective_team, model, device, config, focal_depth + 1)
    child.visit_count += 1
    child.value_sum += value
    node.visit_count += 1
    node.value_sum += value
    return value


def run_mcts(
    env: DraftEnvironment,
    perspective_team: int,
    model: PolicyValueNet,
    device: "torch.device",
    config: AlphaDraftConfig,
    add_root_noise: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    root = SearchNode()
    candidate_indices, priors, context, candidate_matrix, root_value = expand_node(
        root,
        env,
        perspective_team,
        model,
        device,
        config,
        add_root_noise=add_root_noise,
    )

    for _ in range(config.mcts_simulations):
        env_sim = env.clone()
        simulate_search(root, env_sim, perspective_team, model, device, config, focal_depth=0)

    visit_counts = np.asarray(
        [root.children[int(action)].visit_count for action in candidate_indices],
        dtype=np.float32,
    )
    if float(visit_counts.sum()) <= 0:
        visit_probs = priors
    else:
        visit_probs = visit_counts / visit_counts.sum()
    return candidate_indices, visit_probs.astype(np.float32), context, candidate_matrix, float(root_value)


def select_action_from_visits(
    candidate_indices: np.ndarray,
    visit_probs: np.ndarray,
    round_number: int,
    config: AlphaDraftConfig,
    deterministic: bool,
) -> int:
    if deterministic:
        return int(candidate_indices[int(np.argmax(visit_probs))])

    temperature = (
        config.early_round_temperature
        if round_number < config.late_round_start
        else config.late_round_temperature
    )
    adjusted = softmax_numpy(np.log(np.clip(visit_probs, 1e-9, 1.0)), temperature=temperature)
    return int(np.random.choice(candidate_indices, p=adjusted))


def play_self_play_episode(
    season_pool: SeasonPool,
    model: PolicyValueNet,
    device: "torch.device",
    config: AlphaDraftConfig,
    focal_team_index: int,
    training: bool,
) -> EpisodeSummary:
    model.eval()
    env = DraftEnvironment(season_pool, config)
    examples: List[TrainingExample] = []

    while not env.is_terminal():
        current_team = env.current_team()
        if current_team == focal_team_index:
            candidate_indices, visit_probs, context, candidate_matrix, _ = run_mcts(
                env,
                perspective_team=focal_team_index,
                model=model,
                device=device,
                config=config,
                add_root_noise=training,
            )
            action = select_action_from_visits(
                candidate_indices,
                visit_probs,
                round_number=env.current_round(),
                config=config,
                deterministic=not training,
            )
            examples.append(
                TrainingExample(
                    season=season_pool.season,
                    context=context,
                    candidates=candidate_matrix,
                    policy=visit_probs.astype(np.float32),
                    reward=0.0,
                )
            )
        else:
            action = choose_opponent_action(
                env,
                current_team,
                model,
                device,
                config,
                stochastic=training,
            )
        env.step(action)

    team_results, rewards = env.evaluate_terminal()
    focal_row = team_results.loc[team_results["team_index"] == focal_team_index].iloc[0]
    focal_reward = float(rewards[focal_team_index])
    for example in examples:
        example.reward = focal_reward

    return EpisodeSummary(
        season=season_pool.season,
        focal_team_index=focal_team_index,
        focal_points=float(focal_row["total_season_points"]),
        focal_reward=focal_reward,
        focal_rank=int(focal_row["rank"]),
        winner_points=float(team_results["total_season_points"].max()),
        examples=examples,
    )


def play_full_policy_draft(
    season_pool: SeasonPool,
    model: PolicyValueNet,
    device: "torch.device",
    config: AlphaDraftConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    env = DraftEnvironment(season_pool, config)
    while not env.is_terminal():
        current_team = env.current_team()
        candidate_indices, visit_probs, _, _, _ = run_mcts(
            env,
            perspective_team=current_team,
            model=model,
            device=device,
            config=config,
            add_root_noise=False,
        )
        action = select_action_from_visits(
            candidate_indices,
            visit_probs,
            round_number=env.current_round(),
            config=config,
            deterministic=True,
        )
        env.step(action)
    team_results, _ = env.evaluate_terminal()
    team_results = team_results.sort_values("rank").reset_index(drop=True)
    return env.draft_board_frame(), team_results


def build_batch(examples: Sequence[TrainingExample]) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    batch_size = len(examples)
    max_candidates = max(example.candidates.shape[0] for example in examples)
    candidate_dim = examples[0].candidates.shape[1]
    context_dim = examples[0].context.shape[0]

    context = np.zeros((batch_size, context_dim), dtype=np.float32)
    candidates = np.zeros((batch_size, max_candidates, candidate_dim), dtype=np.float32)
    policy = np.zeros((batch_size, max_candidates), dtype=np.float32)
    mask = np.zeros((batch_size, max_candidates), dtype=bool)
    reward = np.zeros(batch_size, dtype=np.float32)

    for row_idx, example in enumerate(examples):
        count = example.candidates.shape[0]
        context[row_idx] = example.context
        candidates[row_idx, :count] = example.candidates
        policy[row_idx, :count] = example.policy
        mask[row_idx, :count] = True
        reward[row_idx] = example.reward

    return (
        torch.from_numpy(context),
        torch.from_numpy(candidates),
        torch.from_numpy(mask),
        torch.from_numpy(policy),
        torch.from_numpy(reward),
    )


def train_on_replay_buffer(
    model: nn.Module,
    optimizer: "torch.optim.Optimizer",
    replay_buffer: ReplayBuffer,
    device: "torch.device",
    config: AlphaDraftConfig,
    iteration: int,
) -> Dict[str, float]:
    if len(replay_buffer) < config.batch_size:
        return {"policy_loss": np.nan, "value_loss": np.nan, "entropy": np.nan}

    model.train()
    examples = replay_buffer.as_list()
    random.shuffle(examples)
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []

    for epoch_idx in range(config.training_epochs_per_iteration):
        epoch_policy_losses: List[float] = []
        epoch_value_losses: List[float] = []
        epoch_entropies: List[float] = []
        for batch_start in range(0, len(examples), config.batch_size):
            batch_examples = examples[batch_start : batch_start + config.batch_size]
            if len(batch_examples) < 2:
                continue
            context, candidates, mask, target_policy, reward = build_batch(batch_examples)
            context = context.to(device)
            candidates = candidates.to(device)
            mask = mask.to(device)
            target_policy = target_policy.to(device)
            reward = reward.to(device)

            logits, values = model(context, candidates, mask)
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            smoothed_policy = (1.0 - config.label_smoothing) * target_policy
            smoothed_policy = smoothed_policy + (config.label_smoothing * mask.float() / valid_counts.float())

            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            policy_loss = -(smoothed_policy * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(values, reward)
            entropy = -(probs * log_probs).sum(dim=1).mean()

            loss = (
                (config.policy_loss_weight * policy_loss)
                + (config.value_loss_weight * value_loss)
                - (config.entropy_bonus_weight * entropy)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            epoch_policy_losses.append(float(policy_loss.item()))
            epoch_value_losses.append(float(value_loss.item()))
            epoch_entropies.append(float(entropy.item()))

        if epoch_policy_losses:
            log_progress(
                "Iteration "
                f"{iteration} train epoch {epoch_idx + 1}/{config.training_epochs_per_iteration} | "
                f"policy_loss={np.nanmean(epoch_policy_losses):.4f} | "
                f"value_loss={np.nanmean(epoch_value_losses):.4f} | "
                f"entropy={np.nanmean(epoch_entropies):.4f}"
            )

    return {
        "policy_loss": float(np.nanmean(policy_losses)) if policy_losses else np.nan,
        "value_loss": float(np.nanmean(value_losses)) if value_losses else np.nan,
        "entropy": float(np.nanmean(entropies)) if entropies else np.nan,
    }


def evaluate_on_seasons(
    season_pools: Dict[int, SeasonPool],
    model: nn.Module,
    device: "torch.device",
    config: AlphaDraftConfig,
    phase_label: str = "validation",
) -> pd.DataFrame:
    rows: List[dict] = []
    model.eval()
    for season, season_pool in sorted(season_pools.items()):
        season_rows: List[dict] = []
        for team_index in range(config.league_size):
            summary = play_self_play_episode(
                season_pool,
                model,
                device,
                config,
                focal_team_index=team_index,
                training=False,
            )
            row = {
                "season": season,
                "team_index": team_index,
                "focal_points": summary.focal_points,
                "focal_reward": summary.focal_reward,
                "focal_rank": summary.focal_rank,
                "winner_points": summary.winner_points,
            }
            season_rows.append(row)
            rows.append(row)
        if season_rows:
            season_df = pd.DataFrame(season_rows)
            log_progress(
                f"{phase_label.title()} season {season} | "
                f"mean_reward={season_df['focal_reward'].mean():.4f} | "
                f"mean_points={season_df['focal_points'].mean():.2f} | "
                f"mean_rank={season_df['focal_rank'].mean():.2f}"
            )
    return pd.DataFrame(rows)


def fit_alpha_self_play_model(
    train_pools: Dict[int, SeasonPool],
    valid_pools: Dict[int, SeasonPool],
    context_dim: int,
    candidate_dim: int,
    config: AlphaDraftConfig,
    device: "torch.device",
) -> Tuple[PolicyValueNet, pd.DataFrame, pd.DataFrame]:
    model = build_training_model(context_dim, candidate_dim, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    history_rows: List[dict] = []
    best_state_dict = None
    best_validation_reward = -np.inf
    stale_iterations = 0
    train_seasons = sorted(train_pools.keys())

    log_progress(
        f"Starting self-play training | train_seasons={train_seasons[0]}-{train_seasons[-1]} | "
        f"validation_seasons={list(valid_pools.keys())} | iterations={config.self_play_iterations} | "
        f"episodes_per_iteration={config.episodes_per_iteration} | replay_buffer={config.replay_buffer_size}"
    )

    for iteration in range(1, config.self_play_iterations + 1):
        log_progress(f"Iteration {iteration}/{config.self_play_iterations} | self-play generation started")
        iteration_examples: List[TrainingExample] = []
        episode_rows: List[dict] = []
        for episode_number in range(1, config.episodes_per_iteration + 1):
            season = int(np.random.choice(train_seasons))
            focal_team_index = int(np.random.randint(0, config.league_size))
            summary = play_self_play_episode(
                train_pools[season],
                model,
                device,
                config,
                focal_team_index=focal_team_index,
                training=True,
            )
            iteration_examples.extend(summary.examples)
            episode_rows.append(
                {
                    "iteration": iteration,
                    "season": season,
                    "team_index": focal_team_index,
                    "focal_points": summary.focal_points,
                    "focal_reward": summary.focal_reward,
                    "focal_rank": summary.focal_rank,
                    "winner_points": summary.winner_points,
                }
            )
            if episode_number % max(config.log_every_episodes, 1) == 0:
                recent_df = pd.DataFrame(episode_rows[-max(config.log_every_episodes, 1) :])
                log_progress(
                    f"Iteration {iteration} self-play {episode_number}/{config.episodes_per_iteration} | "
                    f"recent_reward={recent_df['focal_reward'].mean():.4f} | "
                    f"recent_points={recent_df['focal_points'].mean():.2f} | "
                    f"replay_size={len(replay_buffer) + len(iteration_examples)}"
                )

        replay_buffer.extend(iteration_examples)
        log_progress(
            f"Iteration {iteration} | self-play complete | examples_added={len(iteration_examples)} | "
            f"replay_size={len(replay_buffer)}"
        )
        train_metrics = train_on_replay_buffer(model, optimizer, replay_buffer, device, config, iteration=iteration)
        log_progress(f"Iteration {iteration} | validation started")
        validation_df = evaluate_on_seasons(valid_pools, model, device, config, phase_label=f"validation iter {iteration}")
        validation_reward = float(validation_df["focal_reward"].mean()) if not validation_df.empty else -np.inf
        validation_points = float(validation_df["focal_points"].mean()) if not validation_df.empty else np.nan
        validation_rank = float(validation_df["focal_rank"].mean()) if not validation_df.empty else np.nan
        train_episode_df = pd.DataFrame(episode_rows)
        history_rows.append(
            {
                "iteration": iteration,
                "replay_size": len(replay_buffer),
                "examples_added": len(iteration_examples),
                "train_policy_loss": train_metrics["policy_loss"],
                "train_value_loss": train_metrics["value_loss"],
                "train_entropy": train_metrics["entropy"],
                "self_play_reward_mean": float(train_episode_df["focal_reward"].mean()) if not train_episode_df.empty else np.nan,
                "self_play_points_mean": float(train_episode_df["focal_points"].mean()) if not train_episode_df.empty else np.nan,
                "validation_reward_mean": validation_reward,
                "validation_points_mean": validation_points,
                "validation_rank_mean": validation_rank,
            }
        )
        log_progress(
            f"Iteration {iteration} summary | "
            f"train_policy_loss={train_metrics['policy_loss']:.4f} | "
            f"train_value_loss={train_metrics['value_loss']:.4f} | "
            f"validation_reward={validation_reward:.4f} | "
            f"validation_points={validation_points:.2f} | "
            f"validation_rank={validation_rank:.2f}"
        )

        if validation_reward > best_validation_reward:
            best_validation_reward = validation_reward
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in unwrap_model(model).state_dict().items()
            }
            stale_iterations = 0
            log_progress(f"Iteration {iteration} | new best checkpoint | validation_reward={validation_reward:.4f}")
        else:
            stale_iterations += 1
            if stale_iterations >= config.validation_patience:
                log_progress(
                    f"Early stopping triggered after iteration {iteration} | "
                    f"best_validation_reward={best_validation_reward:.4f}"
                )
                break

    if best_state_dict is not None:
        unwrap_model(model).load_state_dict(best_state_dict)

    history_df = pd.DataFrame(history_rows)
    log_progress("Final validation pass started")
    final_validation_df = evaluate_on_seasons(valid_pools, model, device, config, phase_label="final validation")
    return model, history_df, final_validation_df


def summarize_test_run(team_results: pd.DataFrame) -> dict:
    return {
        "best_team_points": float(team_results["total_season_points"].max()),
        "mean_team_points": float(team_results["total_season_points"].mean()),
        "std_team_points": float(team_results["total_season_points"].std(ddof=0)),
    }


def save_outputs(
    output_dir: Path,
    config: AlphaDraftConfig,
    history_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    draft_board_df: pd.DataFrame,
    team_results_df: pd.DataFrame,
    test_summary: dict,
    model: nn.Module,
    encoder: FeatureEncoder,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    validation_df.to_csv(output_dir / "validation_results.csv", index=False)
    draft_board_df.to_csv(output_dir / "test_2024_draft_board.csv", index=False)
    team_results_df.to_csv(output_dir / "test_2024_team_results.csv", index=False)

    metadata = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "feature_count": len(encoder.feature_names),
        "numeric_features": encoder.numeric_features,
        "categorical_features": encoder.categorical_features,
        "test_summary": test_summary,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "feature_names": encoder.feature_names,
            "numeric_features": encoder.numeric_features,
            "categorical_features": encoder.categorical_features,
        },
        output_dir / "alphadraft_zero_model.pt",
    )


def parse_args() -> AlphaDraftConfig:
    parser = argparse.ArgumentParser(description="AlphaZero-style self-play trainer for fantasy football drafts.")
    parser.add_argument("--csv-path", type=Path, default=Path("fantasy_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_alphadraft_zero"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes-per-iteration", type=int, default=48)
    parser.add_argument("--mcts-simulations", type=int, default=40)
    parser.add_argument("--mcts-focal-depth", type=int, default=2)
    parser.add_argument("--max-players-per-season", type=int, default=220)
    parser.add_argument("--shortlist-size", type=int, default=48)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    return AlphaDraftConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        self_play_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        mcts_simulations=args.mcts_simulations,
        mcts_focal_depth=args.mcts_focal_depth,
        max_players_per_season=args.max_players_per_season,
        shortlist_size=args.shortlist_size,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    device = select_device(config.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    set_random_seed(config.seed)
    gpu_count = get_visible_gpu_count(device)
    if gpu_count > 0:
        gpu_names = [torch.cuda.get_device_name(idx) for idx in range(gpu_count)]
        log_progress(f"Using device: {device} | visible_gpus={gpu_count} | gpu_names={gpu_names}")
        if gpu_count > 1:
            log_progress("Multi-GPU mode enabled with torch.nn.DataParallel")
    else:
        log_progress(f"Using device: {device}")

    log_progress("Loading data and building historical features...")
    raw_df = load_data(config.csv_path)
    feature_df = build_features(raw_df)

    train_end = min(config.validation_seasons) - 1
    train_mask = feature_df["season"].between(config.min_train_season, train_end)
    valid_mask = feature_df["season"].isin(config.validation_seasons)
    test_mask = feature_df["season"].eq(config.test_season)

    numeric_features, categorical_features = get_model_features(feature_df)
    numeric_features = [column for column in numeric_features if column in feature_df.columns]
    categorical_features = [column for column in categorical_features if column in feature_df.columns]

    log_progress("Fitting train-only feature encoder to prevent leakage...")
    encoder = FeatureEncoder(numeric_features, categorical_features)
    encoder.fit(feature_df.loc[train_mask].copy())

    log_progress("Building season pools for self-play, validation, and 2024 test...")
    train_seasons = sorted(feature_df.loc[train_mask, "season"].unique().tolist())
    valid_seasons = sorted(feature_df.loc[valid_mask, "season"].unique().tolist())
    test_seasons = sorted(feature_df.loc[test_mask, "season"].unique().tolist())
    train_pools = build_season_pools(feature_df, encoder, train_seasons, config)
    valid_pools = build_season_pools(feature_df, encoder, valid_seasons, config)
    test_pools = build_season_pools(feature_df, encoder, test_seasons, config)

    if not train_pools or not valid_pools or config.test_season not in test_pools:
        raise SystemExit("Missing train, validation, or 2024 test seasons after pool construction.")

    sample_pool = next(iter(train_pools.values()))
    sample_env = DraftEnvironment(sample_pool, config)
    sample_candidates = sample_env.shortlist(team_index=0, limit=config.shortlist_size)
    context_dim = int(sample_env.build_context_vector(0).shape[0])
    candidate_dim = int(sample_env.build_candidate_matrix(0, sample_candidates).shape[1])

    log_progress(
        f"Season split | train={train_seasons[0]}-{train_seasons[-1]} | "
        f"validation={valid_seasons} | test={test_seasons}"
    )
    log_progress(
        f"Model input sizes | context_dim={context_dim} | candidate_dim={candidate_dim} | "
        f"batch_size={config.batch_size} | shortlist={config.shortlist_size} | mcts_simulations={config.mcts_simulations}"
    )
    log_progress("Training AlphaZero-style policy/value network with self-play...")
    model, history_df, validation_df = fit_alpha_self_play_model(
        train_pools=train_pools,
        valid_pools=valid_pools,
        context_dim=context_dim,
        candidate_dim=candidate_dim,
        config=config,
        device=device,
    )

    log_progress("Running deterministic 2024 policy draft with the best validation checkpoint...")
    draft_board_df, team_results_df = play_full_policy_draft(
        test_pools[config.test_season],
        model,
        device,
        config,
    )
    test_summary = summarize_test_run(team_results_df)

    log_progress("Saving model artifacts and reports...")
    save_outputs(
        output_dir=config.output_dir,
        config=config,
        history_df=history_df,
        validation_df=validation_df,
        draft_board_df=draft_board_df,
        team_results_df=team_results_df,
        test_summary=test_summary,
        model=model,
        encoder=encoder,
    )

    log_progress("Complete.")
    log_progress(
        f"Validation mean reward: {validation_df['focal_reward'].mean():.4f} | "
        f"2024 best team points: {team_results_df['total_season_points'].max():.2f}"
    )


if __name__ == "__main__":
    main()
