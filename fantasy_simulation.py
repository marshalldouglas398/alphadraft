from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
LEAGUE_SIZE = 8
DRAFT_ROUNDS = 15
SCORING_COLUMN = "fantasy_points_total"
REPLACEMENT_RANKS = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
PRIMARY_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX_POSITIONS = {"RB", "WR", "TE"}
FLEX_SLOTS = 1
BENCH_SOFT_CAPS = {"QB": 2, "RB": 6, "WR": 6, "TE": 2}
SCARCITY_LAMBDA = 0.35
STACK_BONUS = 0.07
COMPETITION_PENALTY = 0.08
TOP_CANDIDATES_TO_SAMPLE = 10
SOFTMAX_TEMPERATURE = 0.6
MIN_MODEL_SEASON = 2010
RECENCY_HALFLIFE = 4.0
POSITION_TOP_POOL = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
ALL_TOP_POOL = 100
ROOKIE_PRIOR_QUANTILE = {"QB": 0.80, "RB": 0.78, "WR": 0.76, "TE": 0.72}
ROOKIE_PPG_BLEND_WEIGHTS = {"QB": 0.42, "RB": 0.36, "WR": 0.34, "TE": 0.28}
ROOKIE_GAMES_BLEND_WEIGHTS = {"QB": 0.30, "RB": 0.24, "WR": 0.22, "TE": 0.18}
COMEBACK_PPG_BLEND_WEIGHTS = {"QB": 0.42, "RB": 0.34, "WR": 0.30, "TE": 0.28}
COMEBACK_GAMES_BLEND_WEIGHTS = {"QB": 0.28, "RB": 0.24, "WR": 0.22, "TE": 0.20}
VOR_DRAFT_WEIGHT = 0.08
RESERVE_DRAFT_WEIGHT = 0.12
AVAILABILITY_THRESHOLDS = [5, 9, 13, 16]
AVAILABILITY_BUCKET_MIDS = np.array([2.0, 6.5, 10.5, 14.0, 16.5])
AVAILABILITY_DOWNSIDE_INDEX = 3
ELITE_FINISH_RANKS = {"QB": 3, "RB": 8, "WR": 12, "TE": 3}
TOP_TIER_FINISH_RANKS = {"QB": 6, "RB": 12, "WR": 18, "TE": 6}
BREAKOUT_DELTA = 0.18
MAJOR_JUMP_DELTA = 0.28
COLLAPSE_DELTA = 0.22
ROLE_EXPANSION_SHARE_DELTA = {"QB": 0.06, "RB": 0.07, "WR": 0.06, "TE": 0.05}
ROLE_EXPANSION_VOLUME_GROWTH = {"QB": 0.10, "RB": 0.18, "WR": 0.16, "TE": 0.14}
RERANK_BLEND = 0.30
UPSIDE_BONUS_WEIGHT = 24.0
DOWNSIDE_PENALTY_WEIGHT = 16.0
ROLLOUT_CANDIDATES = 8
ROLLOUT_NEXT_PICK_WEIGHT = 0.72
ROLLOUT_POOL_WEIGHT = 0.35
OPPONENT_PUBLIC_WEIGHT = 0.72
OPPONENT_NEED_WEIGHT = 0.28
POOL_PRESSURE_WEIGHT = 0.12
HIERARCHICAL_SHRINKAGE = 12.0
SHIFT_WEIGHT_CLIP = (0.35, 4.0)
EVENT_CALIBRATION_MIN_SAMPLES = 18
POINT_DISTRIBUTION_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
ROBUST_UPSIDE_WEIGHT = 0.48
ROBUST_DOWNSIDE_WEIGHT = 0.28
SUBMODULAR_ARCHETYPE_PENALTY = 0.12
TEAM_DEPENDENCY_PENALTY = 0.08
RISK_BUCKET_PENALTY = 0.10
SAFE_POLICY_MARGIN = 4.0
SAFE_POLICY_BLEND = 0.90
MIXTURE_EXPERT_SHRINKAGE = 10.0
OPPONENT_TEMPERATURE = 0.78
POINT_CALIBRATION_BLEND = {"QB": 0.82, "RB": 0.84, "WR": 0.84, "TE": 0.90, "ALL": 0.86}
CEILING_CALIBRATION_BLEND = {"QB": 0.88, "RB": 0.90, "WR": 0.90, "TE": 0.94, "ALL": 0.90}
POINT_CALIBRATION_MIN_STD_RATIO = {"QB": 0.78, "RB": 0.74, "WR": 0.74, "TE": 0.68, "ALL": 0.72}
CEILING_CALIBRATION_MIN_STD_RATIO = {"QB": 0.82, "RB": 0.78, "WR": 0.78, "TE": 0.72, "ALL": 0.76}
POINT_CALIBRATION_MIN_TOP_UNIQUE = {"QB": 0.70, "RB": 0.55, "WR": 0.55, "TE": 0.50, "ALL": 0.55}
CEILING_CALIBRATION_MIN_TOP_UNIQUE = {"QB": 0.78, "RB": 0.62, "WR": 0.62, "TE": 0.55, "ALL": 0.62}
CALIBRATION_MAX_UNIQUE_DROP = 0.18
PRIOR_GROUP_COLUMNS = [
    "position",
    "age_bucket",
    "experience_bucket",
    "team_context_bucket",
    "history_bucket",
    "position_archetype",
]
OPPONENT_PROFILE_ALPHAS = {
    "balanced": {"QB": 1.1, "RB": 1.4, "WR": 1.4, "TE": 0.9},
    "elite_qb": {"QB": 2.2, "RB": 1.0, "WR": 1.2, "TE": 0.8},
    "anchor_rb": {"QB": 0.8, "RB": 2.1, "WR": 1.1, "TE": 0.8},
    "wr_value": {"QB": 0.9, "RB": 1.0, "WR": 2.1, "TE": 0.9},
    "te_chaser": {"QB": 0.9, "RB": 1.0, "WR": 1.1, "TE": 1.9},
}
POSITION_UPSIDE_MULTIPLIER = {"QB": 0.92, "RB": 1.28, "WR": 1.22, "TE": 0.90}
POSITION_DOWNSIDE_MULTIPLIER = {"QB": 0.82, "RB": 1.02, "WR": 0.94, "TE": 0.88}
POSITION_CEILING_BLEND = {"QB": 0.45, "RB": 0.72, "WR": 0.34, "TE": 0.42}
POSITION_CEILING_DRAFT_WEIGHT = {"QB": 0.12, "RB": 0.34, "WR": 0.10, "TE": 0.09}
POSITION_PUBLIC_CEILING_WEIGHT = {"QB": 0.10, "RB": 0.24, "WR": 0.08, "TE": 0.07}
POSITION_OPPORTUNITY_BONUS_SCALE = {"QB": 5.5, "RB": 14.0, "WR": 4.5, "TE": 7.0}
WEEKLY_SIM_WEEKS = 17
WEEKLY_DIRICHLET_ALPHA = {"QB": 18.0, "RB": 14.0, "WR": 12.0, "TE": 14.0}
WEEKLY_LOGNORMAL_SIGMA = {"QB": 0.05, "RB": 0.10, "WR": 0.12, "TE": 0.11}
TAIL_METRIC_COLUMNS = [
    "ceiling_top_pool_rank_corr",
    "top_tier_hit_rate",
    "major_jump_hit_rate",
    "role_expansion_hit_rate",
    "actual_breakout_ceiling_avg_rank",
    "elite_precision",
    "elite_recall",
]


@dataclass
class TeamState:
    name: str
    roster: List[dict] = field(default_factory=list)
    manager_profile: str = "balanced"
    position_pref_alpha: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n_rows = len(X)
        p = float(np.clip(self.probability, 0.0, 1.0))
        return np.column_stack([np.full(n_rows, 1.0 - p), np.full(n_rows, p)])


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    df = raw.rename(
        columns={
            "Player": "player_name",
            "Tm": "team",
            "Pos": "position",
            "Age": "age",
            "G": "games_played",
            "GS": "games_started",
            "Key": "player_key",
            "Year": "season",
            "Exp": "experience",
            "Pass_Att": "pass_attempts",
            "Rush_Att": "carries",
            "Rec_Tgt": "targets",
            "Points_half-ppr": "fantasy_points_total",
            "PPG_half-ppr": "fantasy_points_per_game",
            "games_played_pct": "games_played_pct",
            "games_started_pct": "games_started_pct",
            "ProBowl": "pro_bowl",
            "AllPro": "all_pro",
        }
    )
    df = df[df["position"].isin(REPLACEMENT_RANKS)].copy()

    numeric_fill_zero = [
        "pass_attempts",
        "carries",
        "targets",
        "fantasy_points_total",
        "fantasy_points_per_game",
        "games_played_pct",
        "games_started_pct",
        "pro_bowl",
        "all_pro",
        "experience",
        "num_games",
    ]
    for column in numeric_fill_zero:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    df["targets"] = df["targets"].astype(float)
    df["season"] = df["season"].astype(int)
    df["age"] = df["age"].fillna(df.groupby("season")["age"].transform("median")).fillna(df["age"].median())
    df["player_name"] = df["player_name"].fillna("Unknown")
    df["team"] = df["team"].fillna("UNK")

    df["usage_volume"] = np.select(
        [
            df["position"].eq("QB"),
            df["position"].eq("RB"),
            df["position"].isin(["WR", "TE"]),
        ],
        [
            df["pass_attempts"],
            df["carries"],
            df["targets"],
        ],
        default=0.0,
    )
    return df


def compute_vor(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    replacement_rows: List[dict] = []

    for (season, position), group in data.groupby(["season", "position"], sort=True):
        threshold = REPLACEMENT_RANKS[position]
        sorted_group = group.sort_values(SCORING_COLUMN, ascending=False).reset_index(drop=True)
        replacement_idx = min(threshold - 1, len(sorted_group) - 1)
        replacement_points = float(sorted_group.loc[replacement_idx, SCORING_COLUMN])
        replacement_rows.append(
            {
                "season": season,
                "position": position,
                "replacement_points": replacement_points,
            }
        )

    replacement_df = pd.DataFrame(replacement_rows)
    data = data.merge(replacement_df, on=["season", "position"], how="left")
    data["vor"] = data[SCORING_COLUMN] - data["replacement_points"]
    return data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = compute_vor(df)
    data["volume_group"] = np.select(
        [
            data["position"].eq("QB"),
            data["position"].eq("RB"),
            data["position"].isin(["WR", "TE"]),
        ],
        ["QB", "RB", "REC"],
        default="OTHER",
    )

    team_points = (
        data.groupby(["season", "team"], as_index=False)[SCORING_COLUMN]
        .sum()
        .rename(columns={SCORING_COLUMN: "team_total_points"})
    )
    team_points["team_rank"] = (
        team_points.groupby("season")["team_total_points"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    qb_volume = (
        data[data["position"] == "QB"]
        .groupby(["season", "team"], as_index=False)["pass_attempts"]
        .sum()
        .rename(columns={"pass_attempts": "team_qb_attempts"})
    )
    rb_volume = (
        data[data["position"] == "RB"]
        .groupby(["season", "team"], as_index=False)["carries"]
        .sum()
        .rename(columns={"carries": "team_rb_carries"})
    )
    receiver_volume = (
        data[data["position"].isin(["WR", "TE"])]
        .groupby(["season", "team"], as_index=False)["targets"]
        .sum()
        .rename(columns={"targets": "team_receiver_targets"})
    )
    team_group_volume = (
        data.groupby(["season", "team", "volume_group"], as_index=False)["usage_volume"]
        .sum()
        .rename(columns={"usage_volume": "team_group_volume"})
    )
    team_position_volume = (
        data.groupby(["season", "team", "position"], as_index=False)["usage_volume"]
        .sum()
        .rename(columns={"usage_volume": "team_position_volume"})
    )
    previous_team_group_volume = team_group_volume.copy()
    previous_team_group_volume["season"] = previous_team_group_volume["season"] + 1
    previous_team_group_volume = previous_team_group_volume.rename(
        columns={"team_group_volume": "prev_year_team_group_volume"}
    )
    previous_team_position_volume = team_position_volume.copy()
    previous_team_position_volume["season"] = previous_team_position_volume["season"] + 1
    previous_team_position_volume = previous_team_position_volume.rename(
        columns={"team_position_volume": "prev_year_team_position_volume"}
    )

    data = data.merge(team_points, on=["season", "team"], how="left")
    data = data.merge(qb_volume, on=["season", "team"], how="left")
    data = data.merge(rb_volume, on=["season", "team"], how="left")
    data = data.merge(receiver_volume, on=["season", "team"], how="left")
    data = data.merge(previous_team_group_volume, on=["season", "team", "volume_group"], how="left")
    data = data.merge(previous_team_position_volume, on=["season", "team", "position"], how="left")

    usage_denominator = np.select(
        [
            data["position"].eq("QB"),
            data["position"].eq("RB"),
            data["position"].isin(["WR", "TE"]),
        ],
        [
            data["team_qb_attempts"],
            data["team_rb_carries"],
            data["team_receiver_targets"],
        ],
        default=np.nan,
    )
    usage_denominator = np.where(pd.isna(usage_denominator), 0.0, usage_denominator)
    data["usage_share_within_team"] = np.where(
        usage_denominator > 0,
        data["usage_volume"] / usage_denominator,
        0.0,
    )

    previous_team_context = team_points.copy()
    previous_team_context["season"] = previous_team_context["season"] + 1
    previous_team_context = previous_team_context.rename(
        columns={
            "team_total_points": "team_prev_year_total_points",
            "team_rank": "team_prev_year_rank",
        }
    )
    data = data.merge(previous_team_context, on=["season", "team"], how="left")

    for column in ["team_prev_year_total_points", "team_prev_year_rank"]:
        data[column] = data[column].fillna(data[column].median())

    data = data.sort_values(["player_key", "season"]).reset_index(drop=True)
    grouped = data.groupby("player_key", sort=False)
    lagged_team = grouped["team"].shift(1)
    lagged_season = grouped["season"].shift(1)

    data["seasons_played_before"] = grouped.cumcount()
    data["years_since_last_season"] = (data["season"] - lagged_season).fillna(1.0)
    data["same_team_as_last_year"] = (data["team"] == lagged_team).fillna(False).astype(int)
    data["new_team_player"] = ((data["seasons_played_before"] > 0) & (data["same_team_as_last_year"] == 0)).astype(int)

    base_history_cols = [
        "games_played",
        "games_played_pct",
        "fantasy_points_total",
        "fantasy_points_per_game",
        "pass_attempts",
        "carries",
        "targets",
        "usage_volume",
        "usage_share_within_team",
        "team_total_points",
        "team_rank",
        "vor",
        "pro_bowl",
        "all_pro",
    ]

    for column in base_history_cols:
        shifted = grouped[column].shift(1)
        data[f"{column}_lag1"] = shifted
        data[f"{column}_lag2"] = grouped[column].shift(2)
        data[f"{column}_lag3"] = grouped[column].shift(3)
        data[f"{column}_roll2"] = (
            shifted.groupby(data["player_key"]).rolling(2, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        data[f"{column}_roll3"] = (
            shifted.groupby(data["player_key"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    points_shifted = grouped["fantasy_points_total"].shift(1)
    vor_shifted = grouped["vor"].shift(1)
    ppg_shifted = grouped["fantasy_points_per_game"].shift(1)
    prior_season_counts = grouped.cumcount()

    data["career_points_before"] = points_shifted.groupby(data["player_key"]).cumsum()
    data["career_vor_before"] = vor_shifted.groupby(data["player_key"]).cumsum()
    data["career_ppg_before"] = (
        ppg_shifted.groupby(data["player_key"]).cumsum() / prior_season_counts.replace(0, np.nan)
    )
    games_pct_shifted = grouped["games_played_pct"].shift(1)
    data["career_games_played_pct_before"] = (
        games_pct_shifted.groupby(data["player_key"]).cumsum() / prior_season_counts.replace(0, np.nan)
    )

    best_points = grouped["fantasy_points_total"].cummax()
    best_vor = grouped["vor"].cummax()
    best_ppg = grouped["fantasy_points_per_game"].cummax()
    data["historical_best_points_before"] = best_points.groupby(data["player_key"]).shift(1)
    data["historical_best_vor_before"] = best_vor.groupby(data["player_key"]).shift(1)
    data["historical_best_ppg_before"] = best_ppg.groupby(data["player_key"]).shift(1)
    data["returning_usage_from_last_year"] = np.where(
        data["same_team_as_last_year"].eq(1),
        data["usage_volume_lag1"].fillna(0.0),
        0.0,
    )
    returning_group_volume = (
        data.groupby(["season", "team", "volume_group"], as_index=False)["returning_usage_from_last_year"]
        .sum()
        .rename(columns={"returning_usage_from_last_year": "returning_group_volume"})
    )
    returning_position_volume = (
        data.groupby(["season", "team", "position"], as_index=False)["returning_usage_from_last_year"]
        .sum()
        .rename(columns={"returning_usage_from_last_year": "returning_position_volume"})
    )
    data = data.merge(returning_group_volume, on=["season", "team", "volume_group"], how="left")
    data = data.merge(returning_position_volume, on=["season", "team", "position"], how="left")
    data["prev_year_team_group_volume"] = data["prev_year_team_group_volume"].fillna(0.0)
    data["returning_group_volume"] = data["returning_group_volume"].fillna(0.0)
    data["vacated_group_volume"] = (
        data["prev_year_team_group_volume"] - data["returning_group_volume"]
    ).clip(lower=0.0)
    data["vacated_group_share"] = np.where(
        data["prev_year_team_group_volume"] > 0,
        data["vacated_group_volume"] / data["prev_year_team_group_volume"],
        0.0,
    )
    data["prev_year_team_position_volume"] = data["prev_year_team_position_volume"].fillna(0.0)
    data["returning_position_volume"] = data["returning_position_volume"].fillna(0.0)
    data["vacated_position_volume"] = (
        data["prev_year_team_position_volume"] - data["returning_position_volume"]
    ).clip(lower=0.0)
    data["vacated_position_share"] = np.where(
        data["prev_year_team_position_volume"] > 0,
        data["vacated_position_volume"] / data["prev_year_team_position_volume"],
        0.0,
    )
    data["returning_position_share"] = np.where(
        data["prev_year_team_position_volume"] > 0,
        data["returning_position_volume"] / data["prev_year_team_position_volume"],
        0.0,
    )

    data["has_prior_season"] = data["seasons_played_before"].gt(0).astype(int)
    data["is_rookie"] = data["seasons_played_before"].eq(0).astype(int)
    data["is_second_year"] = data["seasons_played_before"].eq(1).astype(int)
    data["prior_year_major_absence"] = (
        data["has_prior_season"].eq(1)
        & (
            data["games_played_lag1"].fillna(0).le(8)
            | data["games_played_pct_lag1"].fillna(0).lt(0.55)
        )
    ).astype(int)
    data["prior_year_lost_season"] = (
        data["has_prior_season"].eq(1) & data["games_played_lag1"].fillna(0).le(4)
    ).astype(int)
    data["injury_return_candidate"] = (
        data["has_prior_season"].eq(1)
        & (data["prior_year_major_absence"].eq(1) | data["years_since_last_season"].gt(1))
    ).astype(int)
    data["age_vs_position_mean"] = data["age"] - data.groupby(["season", "position"])["age"].transform("mean")
    peak_age_map = {"QB": 28.0, "RB": 25.0, "WR": 27.0, "TE": 28.0}
    data["position_peak_age"] = data["position"].map(peak_age_map).astype(float)
    data["age_delta_from_peak"] = data["age"] - data["position_peak_age"]
    data["age_squared"] = data["age"] ** 2
    data["age_delta_squared"] = data["age_delta_from_peak"] ** 2
    data["age_bucket"] = pd.cut(
        data["age"],
        bins=[0, 23, 26, 29, 100],
        labels=["young", "prime", "veteran", "aging"],
        include_lowest=True,
    ).astype(str)
    data["experience_bucket"] = pd.cut(
        data["experience"].fillna(0.0),
        bins=[-1, 0, 2, 5, 100],
        labels=["rookie_exp", "early", "mid", "late"],
        include_lowest=True,
    ).astype(str)
    data["team_context_bucket"] = pd.cut(
        data["team_prev_year_rank"].fillna(data["team_prev_year_rank"].median()),
        bins=[0, 8, 20, 40],
        labels=["top_context", "mid_context", "weak_context"],
        include_lowest=True,
    ).astype(str)
    data["history_bucket"] = np.select(
        [
            data["is_rookie"].eq(1),
            data["has_prior_season"].eq(0),
            data["seasons_played_before"].le(1),
            data["prior_year_major_absence"].eq(1),
        ],
        ["rookie", "thin_history", "young_history", "fragile_history"],
        default="full_history",
    )
    data["position_archetype"] = np.select(
        [
            data["position"].eq("QB") & data["carries_lag1"].fillna(0.0).ge(60.0),
            data["position"].eq("QB"),
            data["position"].eq("RB") & data["carries_lag1"].fillna(0.0).ge(200.0),
            data["position"].eq("RB") & data["targets_lag1"].fillna(0.0).ge(60.0),
            data["position"].eq("RB"),
            data["position"].eq("WR") & data["usage_share_within_team_lag1"].fillna(0.0).ge(0.28),
            data["position"].eq("WR") & data["targets_lag1"].fillna(0.0).le(80.0),
            data["position"].eq("WR"),
            data["position"].eq("TE") & data["targets_lag1"].fillna(0.0).ge(90.0),
            data["position"].eq("TE"),
        ],
        [
            "dual_threat_qb",
            "pocket_qb",
            "bell_cow_rb",
            "receiving_rb",
            "committee_rb",
            "alpha_wr",
            "deep_wr",
            "volume_wr",
            "featured_te",
            "support_te",
        ],
        default="generic",
    )
    data["archetype_usage_score"] = np.select(
        [
            data["position"].eq("QB"),
            data["position"].eq("RB"),
            data["position"].isin(["WR", "TE"]),
        ],
        [
            data["pass_attempts_roll2"].fillna(data["pass_attempts_lag1"]).fillna(0.0),
            data["carries_roll2"].fillna(data["carries_lag1"]).fillna(0.0),
            data["targets_roll2"].fillna(data["targets_lag1"]).fillna(0.0),
        ],
        default=0.0,
    )
    trend_bases = [
        "fantasy_points_total",
        "fantasy_points_per_game",
        "usage_volume",
        "usage_share_within_team",
        "pass_attempts",
        "carries",
        "targets",
        "team_total_points",
    ]
    for base_column in trend_bases:
        lag1 = data[f"{base_column}_lag1"].fillna(0.0)
        lag2 = data[f"{base_column}_lag2"].fillna(0.0)
        lag3 = data[f"{base_column}_lag3"].fillna(0.0)
        data[f"{base_column}_delta_lag1_lag2"] = lag1 - lag2
        data[f"{base_column}_delta_lag2_lag3"] = lag2 - lag3
        data[f"{base_column}_acceleration"] = data[f"{base_column}_delta_lag1_lag2"] - data[f"{base_column}_delta_lag2_lag3"]
        data[f"{base_column}_trend_vs_roll3"] = lag1 - data[f"{base_column}_roll3"].fillna(lag2)

    positive_share_jump = data["usage_share_within_team_delta_lag1_lag2"].clip(lower=0.0)
    positive_volume_jump = data["usage_volume_delta_lag1_lag2"].clip(lower=0.0)
    data["role_growth_signal"] = (
        (1.8 * positive_share_jump)
        + (0.006 * positive_volume_jump)
        + (0.9 * data["vacated_position_share"].fillna(0.0))
        + (0.5 * data["vacated_group_share"].fillna(0.0))
    )
    data["opportunity_shock_score"] = (
        data["role_growth_signal"]
        + (0.22 * data["new_team_player"])
        + (0.18 * data["is_rookie"])
        + (0.12 * data["is_second_year"])
        + (0.16 * data["injury_return_candidate"])
        + (0.10 * data["team_total_points_delta_lag1_lag2"].clip(lower=0.0) / 100.0)
    )
    data["same_role_competition_index"] = (
        data["returning_position_share"].fillna(0.0)
        + (0.50 * data["returning_group_volume"].fillna(0.0) / data["prev_year_team_group_volume"].replace(0.0, np.nan))
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["usage_share_jump_flag"] = data["usage_share_within_team_delta_lag1_lag2"].ge(0.04).astype(int)
    data["usage_volume_jump_flag"] = data["usage_volume_delta_lag1_lag2"].ge(
        np.select(
            [
                data["position"].eq("QB"),
                data["position"].eq("RB"),
                data["position"].isin(["WR", "TE"]),
            ],
            [35.0, 30.0, 18.0],
            default=20.0,
        )
    ).astype(int)

    numeric_generated = [column for column in data.columns if column.endswith(("lag1", "lag2", "lag3", "roll2", "roll3"))]
    numeric_generated += [
        "career_points_before",
        "career_vor_before",
        "career_ppg_before",
        "career_games_played_pct_before",
        "historical_best_points_before",
        "historical_best_vor_before",
        "historical_best_ppg_before",
        "years_since_last_season",
        "same_team_as_last_year",
        "has_prior_season",
        "seasons_played_before",
        "is_rookie",
        "is_second_year",
        "prior_year_major_absence",
        "prior_year_lost_season",
        "injury_return_candidate",
        "age_vs_position_mean",
        "new_team_player",
        "prev_year_team_group_volume",
        "returning_group_volume",
        "vacated_group_volume",
        "vacated_group_share",
        "prev_year_team_position_volume",
        "returning_position_volume",
        "returning_position_share",
        "vacated_position_volume",
        "vacated_position_share",
        "position_peak_age",
        "age_delta_from_peak",
        "age_squared",
        "age_delta_squared",
        "archetype_usage_score",
        "role_growth_signal",
        "opportunity_shock_score",
        "same_role_competition_index",
        "usage_share_jump_flag",
        "usage_volume_jump_flag",
    ]

    for column in numeric_generated:
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)

    return data


def get_model_features(feature_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = [
        "season",
        "age",
        "age_squared",
        "age_delta_from_peak",
        "age_delta_squared",
        "position_peak_age",
        "experience",
        "team_prev_year_total_points",
        "team_prev_year_rank",
        "prev_year_team_group_volume",
        "returning_group_volume",
        "vacated_group_volume",
        "vacated_group_share",
        "prev_year_team_position_volume",
        "returning_position_volume",
        "returning_position_share",
        "vacated_position_volume",
        "vacated_position_share",
        "seasons_played_before",
        "years_since_last_season",
        "same_team_as_last_year",
        "has_prior_season",
        "new_team_player",
        "career_points_before",
        "career_vor_before",
        "career_ppg_before",
        "career_games_played_pct_before",
        "historical_best_points_before",
        "historical_best_vor_before",
        "historical_best_ppg_before",
        "games_played_lag1",
        "games_played_lag2",
        "games_played_lag3",
        "games_played_roll2",
        "games_played_roll3",
        "games_played_pct_lag1",
        "games_played_pct_roll2",
        "games_played_pct_roll3",
        "fantasy_points_total_lag1",
        "fantasy_points_total_lag2",
        "fantasy_points_total_lag3",
        "fantasy_points_total_roll2",
        "fantasy_points_total_roll3",
        "fantasy_points_per_game_lag1",
        "fantasy_points_per_game_lag2",
        "fantasy_points_per_game_lag3",
        "fantasy_points_per_game_roll2",
        "fantasy_points_per_game_roll3",
        "pass_attempts_lag1",
        "pass_attempts_roll2",
        "pass_attempts_roll3",
        "carries_lag1",
        "carries_roll2",
        "carries_roll3",
        "targets_lag1",
        "targets_roll2",
        "targets_roll3",
        "usage_volume_lag1",
        "usage_volume_roll2",
        "usage_volume_roll3",
        "usage_share_within_team_lag1",
        "usage_share_within_team_lag2",
        "usage_share_within_team_roll2",
        "usage_share_within_team_roll3",
        "team_total_points_lag1",
        "team_total_points_roll2",
        "team_total_points_roll3",
        "team_rank_lag1",
        "team_rank_roll2",
        "team_rank_roll3",
        "vor_lag1",
        "vor_lag2",
        "vor_lag3",
        "vor_roll2",
        "vor_roll3",
        "pro_bowl_lag1",
        "all_pro_lag1",
        "is_rookie",
        "is_second_year",
        "prior_year_major_absence",
        "prior_year_lost_season",
        "injury_return_candidate",
        "age_vs_position_mean",
        "archetype_usage_score",
        "fantasy_points_total_delta_lag1_lag2",
        "fantasy_points_total_acceleration",
        "fantasy_points_per_game_delta_lag1_lag2",
        "fantasy_points_per_game_acceleration",
        "usage_volume_delta_lag1_lag2",
        "usage_volume_acceleration",
        "usage_share_within_team_delta_lag1_lag2",
        "usage_share_within_team_acceleration",
        "pass_attempts_delta_lag1_lag2",
        "carries_delta_lag1_lag2",
        "targets_delta_lag1_lag2",
        "team_total_points_delta_lag1_lag2",
        "role_growth_signal",
        "opportunity_shock_score",
        "same_role_competition_index",
        "usage_share_jump_flag",
        "usage_volume_jump_flag",
    ]
    categorical_features = [
        "position",
        "age_bucket",
        "experience_bucket",
        "team_context_bucket",
        "history_bucket",
        "position_archetype",
    ]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan}

    y_true_clean = y_true_array[mask]
    y_pred_clean = y_pred_array[mask]
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    if len(y_true_clean) > 1 and np.std(y_true_clean) > 0 and np.std(y_pred_clean) > 0:
        corr = float(np.corrcoef(y_true_clean, y_pred_clean)[0, 1])
    else:
        corr = np.nan
    return {"rmse": rmse, "mae": mae, "corr": corr}


def get_top_pool_size(scope: str) -> int:
    return POSITION_TOP_POOL.get(scope, ALL_TOP_POOL)


def evaluate_draft_pool(
    eval_df: pd.DataFrame,
    predicted_col: str,
    actual_col: str,
    scope: str,
) -> Dict[str, float]:
    if eval_df.empty:
        return {"top_pool_mae": np.nan, "top_pool_rank_corr": np.nan, "top_pool_hit_rate": np.nan}

    pool_size = min(get_top_pool_size(scope), len(eval_df))
    actual_top = eval_df.nlargest(pool_size, actual_col)
    predicted_top = eval_df.nlargest(pool_size, predicted_col)
    union_keys = pd.Index(actual_top["player_key"]).union(predicted_top["player_key"])
    draft_pool = eval_df[eval_df["player_key"].isin(union_keys)].copy()

    top_pool_mae = float((actual_top[predicted_col] - actual_top[actual_col]).abs().mean())
    hit_rate = float(
        len(set(actual_top["player_key"]).intersection(set(predicted_top["player_key"]))) / max(pool_size, 1)
    )

    if draft_pool.shape[0] > 1:
        draft_pool["actual_rank"] = draft_pool[actual_col].rank(method="average", ascending=False)
        draft_pool["predicted_rank"] = draft_pool[predicted_col].rank(method="average", ascending=False)
        top_pool_rank_corr = evaluate_predictions(
            draft_pool["actual_rank"],
            draft_pool["predicted_rank"].to_numpy(),
        )["corr"]
    else:
        top_pool_rank_corr = np.nan

    return {
        "top_pool_mae": top_pool_mae,
        "top_pool_rank_corr": top_pool_rank_corr,
        "top_pool_hit_rate": hit_rate,
    }


def build_ridge_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", RidgeCV(alphas=np.logspace(-2, 2, 13))),
        ]
    )


def build_histgb_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_depth=4,
                    max_iter=350,
                    min_samples_leaf=18,
                    l2_regularization=0.08,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_logistic_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    C=0.7,
                    max_iter=400,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_histgb_classifier_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=250,
                    min_samples_leaf=20,
                    l2_regularization=0.08,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_model_by_name(
    model_name: str, numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    if model_name == "Ridge":
        return build_ridge_pipeline(numeric_features, categorical_features)
    if model_name == "HistGB":
        return build_histgb_pipeline(numeric_features, categorical_features)
    raise ValueError(f"Unsupported model name: {model_name}")


def build_binary_classifier_by_name(
    model_name: str, numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    if model_name == "Logistic":
        return build_logistic_pipeline(numeric_features, categorical_features)
    if model_name == "HistGBCls":
        return build_histgb_classifier_pipeline(numeric_features, categorical_features)
    raise ValueError(f"Unsupported classifier name: {model_name}")


def predict_binary_probability(model: Pipeline | ConstantProbabilityModel, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    return np.asarray(probabilities[:, 1], dtype=float)


def build_availability_target(games_played: pd.Series, threshold: int) -> pd.Series:
    return games_played.fillna(0.0).ge(threshold).astype(int)


def survival_curve_to_bucket_probabilities(survival_probs: np.ndarray) -> np.ndarray:
    if survival_probs.ndim != 2 or survival_probs.shape[1] != len(AVAILABILITY_THRESHOLDS):
        raise ValueError("survival_probs must be shaped (n_rows, len(AVAILABILITY_THRESHOLDS))")
    survival = np.clip(survival_probs, 0.0, 1.0)
    survival = np.minimum.accumulate(survival, axis=1)
    bucket_probabilities = np.column_stack(
        [
            1.0 - survival[:, 0],
            survival[:, 0] - survival[:, 1],
            survival[:, 1] - survival[:, 2],
            survival[:, 2] - survival[:, 3],
            survival[:, 3],
        ]
    )
    bucket_probabilities = np.clip(bucket_probabilities, 0.0, 1.0)
    bucket_totals = bucket_probabilities.sum(axis=1, keepdims=True)
    bucket_totals = np.where(bucket_totals <= 0, 1.0, bucket_totals)
    return bucket_probabilities / bucket_totals


def expected_games_from_survival(survival_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bucket_probabilities = survival_curve_to_bucket_probabilities(survival_probs)
    expected_games = bucket_probabilities @ AVAILABILITY_BUCKET_MIDS
    downside_probability = bucket_probabilities[:, :AVAILABILITY_DOWNSIDE_INDEX].sum(axis=1)
    return expected_games, downside_probability


def fit_availability_candidate_model_set(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    sample_weights: np.ndarray,
) -> Dict[str, dict]:
    candidate_names = ["Logistic", "HistGBCls"]
    fitted_candidates: Dict[str, dict] = {}
    for candidate_name in candidate_names:
        threshold_models: List[Pipeline | ConstantProbabilityModel] = []
        for threshold in AVAILABILITY_THRESHOLDS:
            target = build_availability_target(train_df["games_played"], threshold)
            if target.nunique() < 2:
                threshold_models.append(ConstantProbabilityModel(probability=float(target.iloc[0])))
                continue
            model = build_binary_classifier_by_name(candidate_name, numeric_features, categorical_features)
            model.fit(train_df[feature_columns], target, model__sample_weight=sample_weights)
            threshold_models.append(model)
        fitted_candidates[candidate_name] = {"threshold_models": threshold_models}
    return fitted_candidates


def predict_availability_distribution(bundle: dict, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold_probabilities = np.column_stack(
        [predict_binary_probability(model, X) for model in bundle["threshold_models"]]
    )
    expected_games, downside_probability = expected_games_from_survival(threshold_probabilities)
    bucket_probabilities = survival_curve_to_bucket_probabilities(threshold_probabilities)
    return expected_games, downside_probability, bucket_probabilities


def predict_availability_bundle(bundle: dict, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    expected_games, downside_probability, _ = predict_availability_distribution(bundle, X)
    return expected_games, downside_probability


def compute_softmax_weights(metric_map: Dict[str, float], higher_is_better: bool) -> Dict[str, float]:
    if not metric_map:
        return {}

    names = list(metric_map.keys())
    values = np.asarray([metric_map[name] for name in names], dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        uniform = 1.0 / len(names)
        return {name: uniform for name in names}

    safe_values = values.copy()
    safe_values[~finite_mask] = np.nanmedian(values[finite_mask])
    centered = safe_values - np.nanmax(safe_values) if higher_is_better else np.nanmin(safe_values) - safe_values
    scale = max(np.nanstd(safe_values), 1e-6)
    weights = np.exp(centered / scale)
    weights = np.clip(weights, 1e-9, None)
    weights = weights / weights.sum()
    return {name: float(weight) for name, weight in zip(names, weights)}


def weighted_average_series(value_map: Dict[str, np.ndarray], weight_map: Dict[str, float]) -> np.ndarray:
    if not value_map:
        return np.array([], dtype=float)

    first_values = np.asarray(next(iter(value_map.values())), dtype=float)
    weighted = np.zeros_like(first_values, dtype=float)
    total_weight = 0.0
    for name, values in value_map.items():
        weight = float(weight_map.get(name, 0.0))
        weighted = weighted + (weight * np.asarray(values, dtype=float))
        total_weight += weight
    if total_weight <= 0:
        total_weight = float(len(value_map))
        weighted = np.zeros_like(first_values, dtype=float)
        for values in value_map.values():
            weighted = weighted + np.asarray(values, dtype=float)
    return weighted / max(total_weight, 1e-9)


def bucket_probabilities_to_quantiles(bucket_probabilities: np.ndarray) -> Dict[float, np.ndarray]:
    quantile_outputs: Dict[float, np.ndarray] = {}
    cumulative = np.cumsum(bucket_probabilities, axis=1)
    for quantile in POINT_DISTRIBUTION_QUANTILES:
        bucket_idx = (cumulative < quantile).sum(axis=1)
        bucket_idx = np.clip(bucket_idx, 0, len(AVAILABILITY_BUCKET_MIDS) - 1)
        quantile_outputs[quantile] = AVAILABILITY_BUCKET_MIDS[bucket_idx]
    return quantile_outputs


def compute_density_ratio_weights(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_columns: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
) -> np.ndarray:
    if source_df.empty or target_df.empty:
        return np.ones(len(source_df), dtype=float)

    domain_train = pd.concat([source_df[feature_columns], target_df[feature_columns]], ignore_index=True)
    domain_target = np.concatenate([np.zeros(len(source_df), dtype=int), np.ones(len(target_df), dtype=int)])
    domain_model = build_logistic_pipeline(numeric_features, categorical_features)
    domain_model.fit(domain_train, domain_target)
    target_probability = predict_binary_probability(domain_model, source_df[feature_columns])
    target_probability = np.clip(target_probability, 1e-4, 1.0 - 1e-4)
    prior_ratio = len(source_df) / max(len(target_df), 1)
    density_ratio = (target_probability / (1.0 - target_probability)) * prior_ratio
    return np.clip(density_ratio, SHIFT_WEIGHT_CLIP[0], SHIFT_WEIGHT_CLIP[1])


def build_probability_subgroups(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                df["is_rookie"].eq(1),
                df["injury_return_candidate"].eq(1),
                df["seasons_played_before"].le(1),
            ],
            ["rookie", "comeback", "thin_history"],
            default="established",
        ),
        index=df.index,
        dtype="object",
    )


def fit_probability_candidate_models(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    target: pd.Series,
    sample_weights: np.ndarray,
) -> Dict[str, Pipeline | ConstantProbabilityModel]:
    fitted_models: Dict[str, Pipeline | ConstantProbabilityModel] = {}
    for candidate_name in ["Logistic", "HistGBCls"]:
        if target.nunique() < 2:
            fitted_models[candidate_name] = ConstantProbabilityModel(probability=float(target.iloc[0]))
            continue
        model = build_binary_classifier_by_name(candidate_name, numeric_features, categorical_features)
        model.fit(train_df[feature_columns], target, model__sample_weight=sample_weights)
        fitted_models[candidate_name] = model
    return fitted_models


def brier_score_loss(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(probs)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((actual[mask] - probs[mask]) ** 2))


def fit_isotonic_model(
    probabilities: np.ndarray,
    target: pd.Series | np.ndarray,
) -> IsotonicRegression | None:
    probs = np.asarray(probabilities, dtype=float)
    actual = np.asarray(target, dtype=float)
    mask = np.isfinite(probs) & np.isfinite(actual)
    if mask.sum() < EVENT_CALIBRATION_MIN_SAMPLES:
        return None
    if np.unique(actual[mask]).size < 2 or np.unique(probs[mask]).size < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(probs[mask], actual[mask])
    return model


def fit_grouped_probability_calibrator(
    valid_df: pd.DataFrame,
    probabilities: np.ndarray,
    target: pd.Series,
) -> dict:
    seasons = sorted(valid_df["season"].dropna().unique().tolist())
    calibration_train_mask = valid_df["season"].eq(seasons[0]) if seasons else pd.Series(True, index=valid_df.index)
    calibration_test_mask = valid_df["season"].eq(seasons[-1]) if len(seasons) > 1 else calibration_train_mask
    subgroup_train = build_probability_subgroups(valid_df.loc[calibration_train_mask])
    subgroup_full = build_probability_subgroups(valid_df)

    global_model = fit_isotonic_model(probabilities[calibration_train_mask.to_numpy()], target.loc[calibration_train_mask])
    subgroup_models: Dict[str, IsotonicRegression] = {}
    for subgroup in subgroup_train.unique():
        subgroup_mask = subgroup_train.eq(subgroup).to_numpy()
        subgroup_model = fit_isotonic_model(
            probabilities[calibration_train_mask.to_numpy()][subgroup_mask],
            target.loc[calibration_train_mask].to_numpy()[subgroup_mask],
        )
        if subgroup_model is not None:
            subgroup_models[str(subgroup)] = subgroup_model

    calibration_bundle = {
        "use_calibration": False,
        "global_model": None,
        "subgroup_models": {},
    }

    if global_model is None and not subgroup_models:
        return calibration_bundle

    raw_test = probabilities[calibration_test_mask.to_numpy()]
    actual_test = target.loc[calibration_test_mask]
    calibrated_test = raw_test.copy()
    subgroup_test = build_probability_subgroups(valid_df.loc[calibration_test_mask]).astype(str)
    for subgroup in subgroup_test.unique():
        subgroup_mask = subgroup_test.eq(subgroup).to_numpy()
        subgroup_model = subgroup_models.get(str(subgroup))
        if subgroup_model is not None:
            calibrated_test[subgroup_mask] = subgroup_model.predict(raw_test[subgroup_mask])
        elif global_model is not None:
            calibrated_test[subgroup_mask] = global_model.predict(raw_test[subgroup_mask])

    raw_brier = brier_score_loss(actual_test, raw_test)
    calibrated_brier = brier_score_loss(actual_test, calibrated_test)
    if pd.notna(calibrated_brier) and pd.notna(raw_brier) and calibrated_brier < raw_brier - 0.002:
        calibration_bundle["use_calibration"] = True
        calibration_bundle["global_model"] = fit_isotonic_model(probabilities, target)
        full_subgroups = subgroup_full.astype(str)
        full_models: Dict[str, IsotonicRegression] = {}
        for subgroup in full_subgroups.unique():
            subgroup_mask = full_subgroups.eq(subgroup).to_numpy()
            subgroup_model = fit_isotonic_model(probabilities[subgroup_mask], target.to_numpy()[subgroup_mask])
            if subgroup_model is not None:
                full_models[str(subgroup)] = subgroup_model
        calibration_bundle["subgroup_models"] = full_models
    return calibration_bundle


def apply_grouped_probability_calibrator(
    probabilities: np.ndarray,
    df: pd.DataFrame,
    calibration_bundle: dict,
) -> np.ndarray:
    calibrated = np.asarray(probabilities, dtype=float).copy()
    if not calibration_bundle or not calibration_bundle.get("use_calibration", False):
        return np.clip(calibrated, 0.0, 1.0)

    subgroups = build_probability_subgroups(df).astype(str)
    global_model = calibration_bundle.get("global_model")
    subgroup_models = calibration_bundle.get("subgroup_models", {})
    for subgroup in subgroups.unique():
        subgroup_mask = subgroups.eq(subgroup).to_numpy()
        subgroup_model = subgroup_models.get(str(subgroup))
        if subgroup_model is not None:
            calibrated[subgroup_mask] = subgroup_model.predict(calibrated[subgroup_mask])
        elif global_model is not None:
            calibrated[subgroup_mask] = global_model.predict(calibrated[subgroup_mask])
    return np.clip(calibrated, 0.0, 1.0)


def fit_availability_bundle(
    model_name: str,
    train_df: pd.DataFrame,
    feature_columns: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    sample_weights: np.ndarray,
) -> dict:
    return fit_availability_candidate_model_set(
        train_df=train_df,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        sample_weights=sample_weights,
    )[model_name]


def get_position_rank_targets(position_df: pd.DataFrame, actual_col: str) -> pd.Series:
    return position_df.groupby("season")[actual_col].rank(method="average", ascending=False)


def get_prior_points_anchor(position_df: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            position_df["historical_best_points_before"],
            position_df["fantasy_points_total_lag1"],
            position_df["fantasy_points_total_roll2"],
            position_df["replacement_points"] + 25.0,
        ],
        axis=1,
    ).max(axis=1, skipna=True).fillna(position_df["replacement_points"] + 25.0)


def get_prior_usage_anchor(position_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    prior_volume_anchor = pd.concat(
        [
            position_df["usage_volume_lag1"],
            position_df["usage_volume_roll2"],
            position_df["usage_volume_roll3"],
        ],
        axis=1,
    ).max(axis=1, skipna=True).fillna(0.0)
    prior_share_anchor = pd.concat(
        [
            position_df["usage_share_within_team_lag1"],
            position_df["usage_share_within_team_roll2"],
            position_df["usage_share_within_team_roll3"],
        ],
        axis=1,
    ).max(axis=1, skipna=True).fillna(0.0)
    return prior_volume_anchor, prior_share_anchor


def build_elite_finish_target(position_df: pd.DataFrame) -> pd.Series:
    elite_cutoff = ELITE_FINISH_RANKS[position_df["position"].iloc[0]]
    ranks = get_position_rank_targets(position_df, actual_col="fantasy_points_total")
    return ranks.le(elite_cutoff).astype(int)


def build_top_tier_finish_target(position_df: pd.DataFrame) -> pd.Series:
    top_tier_cutoff = TOP_TIER_FINISH_RANKS[position_df["position"].iloc[0]]
    ranks = get_position_rank_targets(position_df, actual_col="fantasy_points_total")
    return ranks.le(top_tier_cutoff).astype(int)


def build_major_jump_target(position_df: pd.DataFrame) -> pd.Series:
    prior_anchor = get_prior_points_anchor(position_df)
    ranks = get_position_rank_targets(position_df, actual_col="fantasy_points_total")
    major_jump_cutoff = max(TOP_TIER_FINISH_RANKS[position_df["position"].iloc[0]] + 2, 8)
    return (
        (
            position_df["fantasy_points_total"]
            >= np.maximum(prior_anchor * (1.0 + MAJOR_JUMP_DELTA), position_df["replacement_points"] + 55.0)
        )
        | (ranks.le(major_jump_cutoff) & position_df["is_rookie"].eq(1))
        | (ranks.le(major_jump_cutoff) & position_df["is_second_year"].eq(1))
    ).astype(int)


def build_role_expansion_target(position_df: pd.DataFrame) -> pd.Series:
    position = position_df["position"].iloc[0]
    prior_volume_anchor, prior_share_anchor = get_prior_usage_anchor(position_df)
    share_delta = ROLE_EXPANSION_SHARE_DELTA[position]
    volume_growth = ROLE_EXPANSION_VOLUME_GROWTH[position]
    ranks = get_position_rank_targets(position_df, actual_col="fantasy_points_total")
    role_cutoff = max(TOP_TIER_FINISH_RANKS[position] + 4, 10)
    actual_volume = position_df["usage_volume"].fillna(0.0)
    actual_share = position_df["usage_share_within_team"].fillna(0.0)
    return (
        (
            actual_share
            >= np.maximum(prior_share_anchor + share_delta, prior_share_anchor * (1.0 + (share_delta * 0.55)))
        )
        | (
            actual_volume
            >= np.maximum(prior_volume_anchor * (1.0 + volume_growth), prior_volume_anchor + (volume_growth * 25.0))
        )
        | (position_df["vacated_position_share"].fillna(0.0).ge(0.18) & ranks.le(role_cutoff))
    ).astype(int)


def build_breakout_target(position_df: pd.DataFrame) -> pd.Series:
    return (
        build_top_tier_finish_target(position_df).eq(1)
        | build_major_jump_target(position_df).eq(1)
        | build_role_expansion_target(position_df).eq(1)
    ).astype(int)


def build_ceiling_points_target(position_df: pd.DataFrame) -> pd.Series:
    position = position_df["position"].iloc[0]
    season_max_games = position_df["num_games"].fillna(WEEKLY_SIM_WEEKS).astype(float)
    healthy_case_points = position_df["fantasy_points_per_game"].fillna(0.0) * season_max_games
    prior_anchor = get_prior_points_anchor(position_df)
    actual_points = position_df["fantasy_points_total"].fillna(0.0)
    major_jump = build_major_jump_target(position_df).astype(float)
    top_tier = build_top_tier_finish_target(position_df).astype(float)
    role_expansion = build_role_expansion_target(position_df).astype(float)
    jump_gain = (actual_points - prior_anchor).clip(lower=0.0)
    opportunity_signal = (
        (0.45 * position_df["role_growth_signal"].fillna(0.0))
        + (0.35 * position_df["opportunity_shock_score"].fillna(0.0))
        + (0.25 * position_df["vacated_position_share"].fillna(0.0))
        - (0.20 * position_df["same_role_competition_index"].fillna(0.0))
    ).clip(lower=0.0)
    opportunity_bonus_scale = POSITION_OPPORTUNITY_BONUS_SCALE[position]
    archetype_bonus = np.select(
        [
            position_df["position_archetype"].eq("dual_threat_qb"),
            position_df["position_archetype"].eq("bell_cow_rb"),
            position_df["position_archetype"].eq("receiving_rb"),
            position_df["position_archetype"].eq("alpha_wr"),
            position_df["position_archetype"].eq("featured_te"),
        ],
        [7.0, 10.0, 6.0, 2.5, 4.0],
        default=0.0,
    )
    jump_gain_weight = {"QB": 0.26, "RB": 0.42, "WR": 0.24, "TE": 0.30}[position]
    top_tier_bonus = {"QB": 11.0, "RB": 14.0, "WR": 9.0, "TE": 10.0}[position]
    major_jump_bonus = {"QB": 9.0, "RB": 12.0, "WR": 7.0, "TE": 8.0}[position]
    role_bonus = {"QB": 5.0, "RB": 8.0, "WR": 4.0, "TE": 6.0}[position]
    healthy_blend = {"QB": (0.74, 0.26), "RB": (0.68, 0.32), "WR": (0.76, 0.24), "TE": (0.73, 0.27)}[position]
    ceiling_target = np.maximum(actual_points, (healthy_blend[0] * actual_points) + (healthy_blend[1] * healthy_case_points))
    ceiling_target = (
        ceiling_target
        + (jump_gain_weight * jump_gain)
        + (top_tier_bonus * top_tier)
        + (major_jump_bonus * major_jump)
        + (role_bonus * role_expansion)
        + (opportunity_bonus_scale * opportunity_signal)
        + archetype_bonus
    )
    return ceiling_target.clip(lower=actual_points)


def build_collapse_target(position_df: pd.DataFrame) -> pd.Series:
    prior_anchor = get_prior_points_anchor(position_df)
    return (
        position_df["has_prior_season"].eq(1)
        & (
            position_df["fantasy_points_total"].le(position_df["replacement_points"])
            | position_df["fantasy_points_total"].le(prior_anchor * (1.0 - COLLAPSE_DELTA))
        )
    ).astype(int)


def fit_probability_model(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    target: pd.Series,
    sample_weights: np.ndarray,
) -> Pipeline | ConstantProbabilityModel:
    if target.nunique() < 2:
        return ConstantProbabilityModel(probability=float(target.iloc[0]))

    model = build_logistic_pipeline(numeric_features, categorical_features)
    model.fit(train_df[feature_columns], target, model__sample_weight=sample_weights)
    return model


def build_ranker_target(position_df: pd.DataFrame) -> pd.Series:
    pool_size = POSITION_TOP_POOL[position_df["position"].iloc[0]]
    ranks = get_position_rank_targets(position_df, actual_col="fantasy_points_total")
    gains = np.where(ranks <= pool_size, 1.0 / np.log2(ranks + 1.0), 0.0)
    return pd.Series(gains, index=position_df.index, dtype=float)


def build_augmented_feature_frame(
    base_df: pd.DataFrame,
    feature_columns: List[str],
    extra_columns: List[str],
) -> pd.DataFrame:
    augmented_df = base_df[feature_columns].copy()
    for column in extra_columns:
        augmented_df[column] = base_df[column] if column in base_df.columns else 0.0
    return augmented_df


def build_ranker_feature_frame(base_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    return build_augmented_feature_frame(
        base_df,
        feature_columns,
        [
            "predicted_games",
            "predicted_ppg",
            "predicted_points",
            "predicted_vor",
            "predicted_ceiling_points",
            "elite_finish_probability",
            "top_tier_finish_probability",
            "major_jump_probability",
            "role_expansion_probability",
            "breakout_probability",
            "collapse_probability",
            "availability_downside_probability",
        ],
    )


def build_ceiling_feature_frame(base_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    return build_augmented_feature_frame(
        base_df,
        feature_columns,
        [
            "predicted_games",
            "predicted_ppg",
            "predicted_points",
            "predicted_vor",
            "elite_finish_probability",
            "top_tier_finish_probability",
            "major_jump_probability",
            "role_expansion_probability",
            "breakout_probability",
            "collapse_probability",
            "availability_downside_probability",
        ],
    )


def get_nan_tail_metrics() -> Dict[str, float]:
    return {column: np.nan for column in TAIL_METRIC_COLUMNS}


def evaluate_single_scope_tail_metrics(eval_df: pd.DataFrame, position: str) -> Dict[str, float]:
    if eval_df.empty or "predicted_ceiling_points" not in eval_df.columns:
        return get_nan_tail_metrics()

    working = eval_df.copy()
    working["actual_top_tier"] = build_top_tier_finish_target(working)
    working["actual_major_jump"] = build_major_jump_target(working)
    working["actual_role_expansion"] = build_role_expansion_target(working)
    working["predicted_ceiling_rank"] = working.groupby("season")["predicted_ceiling_points"].rank(
        method="average", ascending=False
    )

    top_tier_hits: List[float] = []
    major_jump_hits: List[float] = []
    role_expansion_hits: List[float] = []
    for _, season_df in working.groupby("season"):
        top_cutoff = min(TOP_TIER_FINISH_RANKS[position], len(season_df))
        if top_cutoff > 0:
            actual_top = set(season_df.nlargest(top_cutoff, "fantasy_points_total")["player_key"])
            predicted_top = set(season_df.nlargest(top_cutoff, "predicted_ceiling_points")["player_key"])
            top_tier_hits.append(len(actual_top.intersection(predicted_top)) / top_cutoff)

        actual_major_keys = set(season_df.loc[season_df["actual_major_jump"].eq(1), "player_key"])
        if actual_major_keys:
            predicted_major = set(
                season_df.nlargest(len(actual_major_keys), "major_jump_probability")["player_key"]
            )
            major_jump_hits.append(len(actual_major_keys.intersection(predicted_major)) / len(actual_major_keys))

        actual_role_keys = set(season_df.loc[season_df["actual_role_expansion"].eq(1), "player_key"])
        if actual_role_keys:
            predicted_role = set(
                season_df.nlargest(len(actual_role_keys), "role_expansion_probability")["player_key"]
            )
            role_expansion_hits.append(len(actual_role_keys.intersection(predicted_role)) / len(actual_role_keys))

    actual_top_tier = working["actual_top_tier"].to_numpy(dtype=int)
    if actual_top_tier.sum() > 0:
        prevalence = float(actual_top_tier.mean())
        threshold = float(
            np.quantile(working["top_tier_finish_probability"].to_numpy(dtype=float), max(0.0, 1.0 - prevalence))
        )
        predicted_elite = working["top_tier_finish_probability"].ge(threshold).to_numpy(dtype=int)
        true_positive = int(np.sum((predicted_elite == 1) & (actual_top_tier == 1)))
        elite_precision = true_positive / max(int(predicted_elite.sum()), 1)
        elite_recall = true_positive / max(int(actual_top_tier.sum()), 1)
    else:
        elite_precision = np.nan
        elite_recall = np.nan

    breakout_mask = (
        working["actual_top_tier"].eq(1) | working["actual_major_jump"].eq(1) | working["actual_role_expansion"].eq(1)
    )
    breakout_avg_rank = (
        float(working.loc[breakout_mask, "predicted_ceiling_rank"].mean()) if breakout_mask.any() else np.nan
    )
    ceiling_pool_metrics = evaluate_draft_pool(
        working,
        predicted_col="predicted_ceiling_points",
        actual_col="fantasy_points_total",
        scope=position,
    )
    return {
        "ceiling_top_pool_rank_corr": ceiling_pool_metrics["top_pool_rank_corr"],
        "top_tier_hit_rate": float(np.mean(top_tier_hits)) if top_tier_hits else np.nan,
        "major_jump_hit_rate": float(np.mean(major_jump_hits)) if major_jump_hits else np.nan,
        "role_expansion_hit_rate": float(np.mean(role_expansion_hits)) if role_expansion_hits else np.nan,
        "actual_breakout_ceiling_avg_rank": breakout_avg_rank,
        "elite_precision": elite_precision,
        "elite_recall": elite_recall,
    }


def evaluate_tail_metrics(eval_df: pd.DataFrame, scope: str) -> Dict[str, float]:
    if eval_df.empty:
        return get_nan_tail_metrics()

    if scope != "ALL":
        return evaluate_single_scope_tail_metrics(eval_df.copy(), scope)

    weighted_metrics: Dict[str, List[Tuple[float, float]]] = {column: [] for column in TAIL_METRIC_COLUMNS}
    for position in REPLACEMENT_RANKS:
        position_df = eval_df[eval_df["position"] == position].copy()
        position_metrics = evaluate_single_scope_tail_metrics(position_df, position)
        weight = float(len(position_df))
        for column, value in position_metrics.items():
            if pd.notna(value):
                weighted_metrics[column].append((weight, float(value)))

    aggregated: Dict[str, float] = {}
    for column, weighted_values in weighted_metrics.items():
        if not weighted_values:
            aggregated[column] = np.nan
            continue
        total_weight = sum(weight for weight, _ in weighted_values)
        aggregated[column] = float(sum(weight * value for weight, value in weighted_values) / max(total_weight, 1.0))
    return aggregated


def standardize_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    std = array.std(ddof=0)
    if std <= 1e-9:
        return np.zeros_like(array)
    return (array - array.mean()) / std


def get_ranker_bonus_scale(position_df: pd.DataFrame) -> float:
    score_spread = float(position_df["fantasy_points_total"].std(ddof=0))
    return max(score_spread * RERANK_BLEND, 8.0)


def compute_recency_weights(train_df: pd.DataFrame) -> np.ndarray:
    max_season = float(train_df["season"].max())
    season_gap = max_season - train_df["season"].astype(float)
    recency = np.power(0.5, season_gap / RECENCY_HALFLIFE)
    relevance = 1.0 + np.clip(train_df["vor"].fillna(0.0), 0.0, None) / 150.0
    return (recency * relevance).to_numpy()


def fit_ridge_pipeline(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    numeric_features: List[str],
    categorical_features: List[str],
    sample_weights: np.ndarray,
) -> Pipeline:
    model = build_ridge_pipeline(numeric_features, categorical_features)
    model.fit(train_df[feature_columns], train_df[target_column], model__sample_weight=sample_weights)
    return model


def fit_model(
    model: Pipeline,
    train_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    sample_weights: np.ndarray,
) -> Pipeline:
    model.fit(train_df[feature_columns], train_df[target_column], model__sample_weight=sample_weights)
    return model


def fit_candidate_model_set(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    numeric_features: List[str],
    categorical_features: List[str],
    sample_weights: np.ndarray,
) -> Dict[str, Pipeline]:
    candidates = {
        "Ridge": build_ridge_pipeline(numeric_features, categorical_features),
        "HistGB": build_histgb_pipeline(numeric_features, categorical_features),
    }
    fitted_models: Dict[str, Pipeline] = {}
    for model_name, model in candidates.items():
        model.fit(train_df[feature_columns], train_df[target_column], model__sample_weight=sample_weights)
        fitted_models[model_name] = model
    return fitted_models


def build_projection_priors(history_df: pd.DataFrame) -> Dict[str, object]:
    priors: Dict[str, object] = {"position_defaults": {}, "group_table": pd.DataFrame()}
    position_defaults: Dict[str, Dict[str, float]] = {}
    grouped_rows: List[dict] = []

    for position in REPLACEMENT_RANKS:
        position_history = history_df[history_df["position"] == position].copy()
        rookie_history = position_history[position_history["is_rookie"] == 1].copy()
        rookie_ppg = rookie_history["fantasy_points_per_game"].dropna()
        rookie_games_pct = rookie_history["games_played_pct"].dropna()
        position_ppg_mean = float(position_history["fantasy_points_per_game"].mean())
        position_games_pct_mean = float(position_history["games_played_pct"].mean())

        if rookie_ppg.empty:
            rookie_ppg_anchor = float(position_history["fantasy_points_per_game"].quantile(0.65))
        else:
            rookie_ppg_anchor = float(rookie_ppg.quantile(ROOKIE_PRIOR_QUANTILE[position]))

        if rookie_games_pct.empty:
            rookie_games_pct_anchor = float(position_history["games_played_pct"].quantile(0.60))
        else:
            rookie_games_pct_anchor = float(rookie_games_pct.quantile(0.65))

        position_defaults[position] = {
            "position_ppg_mean": position_ppg_mean,
            "position_games_pct_mean": position_games_pct_mean,
            "rookie_ppg_anchor": rookie_ppg_anchor,
            "rookie_games_pct_anchor": rookie_games_pct_anchor,
        }

    grouped = (
        history_df.groupby(PRIOR_GROUP_COLUMNS, dropna=False)
        .agg(
            group_count=("player_key", "size"),
            group_ppg_mean=("fantasy_points_per_game", "mean"),
            group_games_pct_mean=("games_played_pct", "mean"),
        )
        .reset_index()
    )

    for _, row in grouped.iterrows():
        position_defaults_row = position_defaults[row["position"]]
        shrink = float(row["group_count"]) / (float(row["group_count"]) + HIERARCHICAL_SHRINKAGE)
        grouped_rows.append(
            {
                **{column: row[column] for column in PRIOR_GROUP_COLUMNS},
                "group_count": float(row["group_count"]),
                "ppg_anchor": (shrink * float(row["group_ppg_mean"]))
                + ((1.0 - shrink) * position_defaults_row["position_ppg_mean"]),
                "games_pct_anchor": (shrink * float(row["group_games_pct_mean"]))
                + ((1.0 - shrink) * position_defaults_row["position_games_pct_mean"]),
            }
        )

    priors["position_defaults"] = position_defaults
    priors["group_table"] = pd.DataFrame(grouped_rows)
    return priors


def apply_projection_priors(
    target_df: pd.DataFrame,
    predicted_games: np.ndarray,
    predicted_ppg: np.ndarray,
    projection_priors: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    games = pd.Series(np.asarray(predicted_games, dtype=float), index=target_df.index).clip(lower=0.0)
    max_games = target_df["num_games"].fillna(17).astype(float)
    games = np.minimum(games, max_games)
    ppg = pd.Series(np.asarray(predicted_ppg, dtype=float), index=target_df.index).clip(lower=0.0)

    max_team_rank = float(target_df["team_prev_year_rank"].fillna(1.0).max())
    normalized_team_context = 1.0 - (
        (target_df["team_prev_year_rank"].fillna(max_team_rank) - 1.0) / max(max_team_rank - 1.0, 1.0)
    )
    vacated_share = target_df["vacated_group_share"].fillna(0.0).clip(lower=0.0, upper=1.0)

    group_table = projection_priors.get("group_table", pd.DataFrame())
    position_defaults = projection_priors.get("position_defaults", {})
    if isinstance(group_table, pd.DataFrame) and not group_table.empty:
        prior_frame = (
            target_df[PRIOR_GROUP_COLUMNS]
            .reset_index()
            .merge(group_table, on=PRIOR_GROUP_COLUMNS, how="left")
            .set_index("index")
            .reindex(target_df.index)
        )
    else:
        prior_frame = pd.DataFrame(index=target_df.index)

    base_ppg_anchor = target_df["position"].map(
        {position: prior["position_ppg_mean"] for position, prior in position_defaults.items()}
    ).fillna(target_df["fantasy_points_per_game_roll3"].fillna(target_df["fantasy_points_per_game_lag1"]).fillna(0.0))
    base_games_pct_anchor = target_df["position"].map(
        {position: prior["position_games_pct_mean"] for position, prior in position_defaults.items()}
    ).fillna(0.70)
    grouped_ppg_anchor = prior_frame.get("ppg_anchor", base_ppg_anchor).fillna(base_ppg_anchor)
    grouped_games_pct_anchor = prior_frame.get("games_pct_anchor", base_games_pct_anchor).fillna(base_games_pct_anchor)
    grouped_count = prior_frame.get("group_count", pd.Series(0.0, index=target_df.index)).fillna(0.0)
    hierarchical_strength = (HIERARCHICAL_SHRINKAGE / (HIERARCHICAL_SHRINKAGE + grouped_count)).clip(0.12, 1.0)

    rookie_mask = target_df["is_rookie"].eq(1)
    rookie_ppg_anchor = target_df["position"].map(
        {position: prior["rookie_ppg_anchor"] for position, prior in position_defaults.items()}
    ).fillna(0.0)
    rookie_games_pct_anchor = target_df["position"].map(
        {position: prior["rookie_games_pct_anchor"] for position, prior in position_defaults.items()}
    ).fillna(0.60)
    rookie_ppg_anchor = grouped_ppg_anchor.combine(rookie_ppg_anchor, np.maximum)
    rookie_ppg_anchor = rookie_ppg_anchor * (0.92 + (0.42 * vacated_share) + (0.10 * normalized_team_context))
    rookie_games_anchor = max_games * (
        grouped_games_pct_anchor.combine(rookie_games_pct_anchor, np.maximum)
        * (0.95 + (0.20 * vacated_share) + (0.05 * normalized_team_context))
    ).clip(lower=0.45, upper=1.0)
    rookie_ppg_weight = target_df["position"].map(ROOKIE_PPG_BLEND_WEIGHTS).fillna(0.30) * (
        0.85 + (0.45 * vacated_share)
    )
    rookie_games_weight = target_df["position"].map(ROOKIE_GAMES_BLEND_WEIGHTS).fillna(0.22) * (
        0.90 + (0.35 * vacated_share)
    )
    rookie_ppg_weight = rookie_ppg_weight * (0.85 + (0.50 * hierarchical_strength))
    rookie_games_weight = rookie_games_weight * (0.85 + (0.45 * hierarchical_strength))

    rookie_ppg_target = ((1.0 - rookie_ppg_weight) * ppg) + (rookie_ppg_weight * rookie_ppg_anchor)
    rookie_games_target = ((1.0 - rookie_games_weight) * games) + (rookie_games_weight * rookie_games_anchor)
    rookie_ppg_adjustment = (np.maximum(ppg, rookie_ppg_target) - ppg).where(rookie_mask, 0.0)
    rookie_games_adjustment = (np.maximum(games, rookie_games_target) - games).where(rookie_mask, 0.0)
    ppg = np.maximum(ppg, rookie_ppg_target).where(rookie_mask, ppg)
    games = np.maximum(games, rookie_games_target).where(rookie_mask, games)

    comeback_mask = target_df["injury_return_candidate"].eq(1)
    age_discount = np.clip(
        1.0 - (0.030 * np.maximum(target_df["age_delta_from_peak"].fillna(0.0), 0.0)),
        0.72,
        1.02,
    )
    comeback_ppg_anchor = pd.concat(
        [
            grouped_ppg_anchor,
            target_df[
        [
            "historical_best_ppg_before",
            "fantasy_points_per_game_lag2",
            "fantasy_points_per_game_roll3",
        ]
            ],
        ],
        axis=1,
    ).max(axis=1, skipna=True).fillna(ppg)
    comeback_ppg_anchor = comeback_ppg_anchor * age_discount
    comeback_games_anchor = pd.concat(
        [
            (max_games * grouped_games_pct_anchor),
            (max_games * target_df["career_games_played_pct_before"].fillna(target_df["games_played_pct_roll3"]).fillna(0.7)),
            target_df["games_played_roll3"],
            target_df["games_played_lag2"],
        ],
        axis=1,
    ).max(axis=1, skipna=True).fillna(games)
    comeback_ppg_weight = target_df["position"].map(COMEBACK_PPG_BLEND_WEIGHTS).fillna(0.28)
    comeback_games_weight = target_df["position"].map(COMEBACK_GAMES_BLEND_WEIGHTS).fillna(0.22)
    comeback_ppg_weight = comeback_ppg_weight * np.where(target_df["prior_year_lost_season"].eq(1), 1.12, 1.0)
    comeback_games_weight = comeback_games_weight * np.where(target_df["prior_year_lost_season"].eq(1), 1.10, 1.0)
    comeback_ppg_weight = comeback_ppg_weight * (0.80 + (0.45 * hierarchical_strength))
    comeback_games_weight = comeback_games_weight * (0.80 + (0.40 * hierarchical_strength))
    comeback_ppg_target = ((1.0 - comeback_ppg_weight) * ppg) + (comeback_ppg_weight * comeback_ppg_anchor)
    comeback_games_target = ((1.0 - comeback_games_weight) * games) + (comeback_games_weight * comeback_games_anchor)
    comeback_ppg_adjustment = (np.maximum(ppg, comeback_ppg_target) - ppg).where(comeback_mask, 0.0)
    comeback_games_adjustment = (np.maximum(games, comeback_games_target) - games).where(comeback_mask, 0.0)
    ppg = np.maximum(ppg, comeback_ppg_target).where(comeback_mask, ppg)
    games = np.maximum(games, comeback_games_target).where(comeback_mask, games)

    thin_history_mask = (
        target_df["is_rookie"].eq(1)
        | target_df["seasons_played_before"].le(1)
        | target_df["new_team_player"].eq(1)
    )
    thin_ppg_weight = 0.12 + (0.20 * hierarchical_strength) + (0.10 * vacated_share)
    thin_games_weight = 0.10 + (0.18 * hierarchical_strength)
    thin_ppg_target = ((1.0 - thin_ppg_weight) * ppg) + (thin_ppg_weight * grouped_ppg_anchor)
    thin_games_target = ((1.0 - thin_games_weight) * games) + (
        thin_games_weight * (max_games * grouped_games_pct_anchor)
    )
    thin_ppg_adjustment = (thin_ppg_target - ppg).where(thin_history_mask, 0.0)
    thin_games_adjustment = (thin_games_target - games).where(thin_history_mask, 0.0)
    ppg = thin_ppg_target.where(thin_history_mask, ppg)
    games = thin_games_target.where(thin_history_mask, games)
    games = np.minimum(games, max_games)

    adjustment_columns = {
        "rookie_games_adjustment": rookie_games_adjustment.to_numpy(),
        "rookie_ppg_adjustment": rookie_ppg_adjustment.to_numpy(),
        "comeback_games_adjustment": comeback_games_adjustment.to_numpy(),
        "comeback_ppg_adjustment": comeback_ppg_adjustment.to_numpy(),
        "thin_history_games_adjustment": thin_games_adjustment.to_numpy(),
        "thin_history_ppg_adjustment": thin_ppg_adjustment.to_numpy(),
    }
    return games.to_numpy(), ppg.to_numpy(), adjustment_columns


def fit_linear_calibration(predicted: pd.Series, actual: pd.Series, weights: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(actual, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) == 0:
        return 1.0, 0.0

    slope, intercept = np.polyfit(x[mask], y[mask], 1, w=np.asarray(weights, dtype=float)[mask])
    slope = float(np.clip(slope, 0.85, 1.15))
    intercept = float(np.clip(intercept, -25.0, 25.0))
    return slope, intercept


def get_calibration_weights(eval_df: pd.DataFrame, scope: str, actual_col: str) -> np.ndarray:
    pool_size = max(get_top_pool_size(scope), 1)
    ranks = eval_df[actual_col].rank(method="dense", ascending=False)
    weights = 1.0 + (pool_size / ranks.clip(lower=1.0))
    return np.clip(weights.to_numpy(), 1.0, 5.0)


def decide_tail_calibration(position_valid_df: pd.DataFrame) -> Tuple[float, float, bool]:
    calibration_train = position_valid_df[position_valid_df["season"] == 2022].copy()
    calibration_test = position_valid_df[position_valid_df["season"] == 2023].copy()
    if calibration_train.empty or calibration_test.empty:
        return 1.0, 0.0, False

    weights = get_calibration_weights(calibration_train, position_valid_df["position"].iloc[0], "fantasy_points_total")
    slope, intercept = fit_linear_calibration(
        calibration_train["predicted_points_uncalibrated"],
        calibration_train["fantasy_points_total"],
        weights=weights,
    )
    raw_metrics = evaluate_draft_pool(
        calibration_test,
        predicted_col="predicted_points_uncalibrated",
        actual_col="fantasy_points_total",
        scope=position_valid_df["position"].iloc[0],
    )
    calibrated_test = calibration_test.copy()
    calibrated_test["predicted_points_calibrated"] = (
        intercept + (slope * calibrated_test["predicted_points_uncalibrated"])
    ).clip(lower=0.0)
    calibrated_metrics = evaluate_draft_pool(
        calibrated_test,
        predicted_col="predicted_points_calibrated",
        actual_col="fantasy_points_total",
        scope=position_valid_df["position"].iloc[0],
    )
    raw_point_mae = evaluate_predictions(
        calibration_test["fantasy_points_total"],
        calibration_test["predicted_points_uncalibrated"].to_numpy(),
    )["mae"]
    calibrated_point_mae = evaluate_predictions(
        calibration_test["fantasy_points_total"],
        calibrated_test["predicted_points_calibrated"].to_numpy(),
    )["mae"]
    use_calibration = (
        (
            calibrated_metrics["top_pool_rank_corr"] > raw_metrics["top_pool_rank_corr"] + 0.01
            or calibrated_metrics["top_pool_hit_rate"] > raw_metrics["top_pool_hit_rate"] + 0.02
        )
        and calibrated_metrics["top_pool_mae"] <= raw_metrics["top_pool_mae"] * 1.02
        and calibrated_point_mae <= raw_point_mae * 1.03
    )
    return slope, intercept, use_calibration


def compute_model_selection_key(
    point_metrics: Dict[str, float],
    pool_metrics: Dict[str, float],
) -> Tuple[float, float, float, float]:
    return (
        pool_metrics["top_pool_mae"],
        -pool_metrics["top_pool_hit_rate"],
        -(pool_metrics["top_pool_rank_corr"] if pd.notna(pool_metrics["top_pool_rank_corr"]) else -1.0),
        point_metrics["mae"],
    )


def compute_model_loss(point_metrics: Dict[str, float], pool_metrics: Dict[str, float]) -> float:
    rank_corr = pool_metrics["top_pool_rank_corr"] if pd.notna(pool_metrics["top_pool_rank_corr"]) else 0.0
    return float(
        pool_metrics["top_pool_mae"]
        + (14.0 * (1.0 - pool_metrics["top_pool_hit_rate"]))
        + (10.0 * (1.0 - max(rank_corr, 0.0)))
        + (0.20 * point_metrics["mae"])
    )


def fit_regression_isotonic_model(
    predicted: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    weights: np.ndarray | None = None,
) -> IsotonicRegression | None:
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(actual, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < EVENT_CALIBRATION_MIN_SAMPLES:
        return None
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip")
    if weights is not None:
        weight_array = np.asarray(weights, dtype=float)
        model.fit(x[mask], y[mask], sample_weight=weight_array[mask])
    else:
        model.fit(x[mask], y[mask])
    return model


def apply_regression_calibration_bundle(values: pd.Series | np.ndarray, calibration_bundle: dict) -> np.ndarray:
    raw_values = np.asarray(values, dtype=float).copy()
    calibrated = raw_values.copy()
    if not calibration_bundle:
        return np.clip(calibrated, 0.0, None)

    method = calibration_bundle.get("method", "identity")
    if method == "linear":
        slope = float(calibration_bundle.get("slope", 1.0))
        intercept = float(calibration_bundle.get("intercept", 0.0))
        calibrated = intercept + (slope * calibrated)
    elif method == "isotonic":
        isotonic_model = calibration_bundle.get("model")
        if isotonic_model is not None:
            calibrated = isotonic_model.predict(calibrated)
    blend = float(calibration_bundle.get("blend", 1.0))
    if blend < 1.0:
        calibrated = (blend * np.asarray(calibrated, dtype=float)) + ((1.0 - blend) * raw_values)
    return np.clip(np.asarray(calibrated, dtype=float), 0.0, None)


def compute_tail_calibration_loss(tail_metrics: Dict[str, float], scope: str) -> float:
    ceiling_rank_corr = (
        tail_metrics["ceiling_top_pool_rank_corr"] if pd.notna(tail_metrics["ceiling_top_pool_rank_corr"]) else 0.0
    )
    top_tier_hit = tail_metrics["top_tier_hit_rate"] if pd.notna(tail_metrics["top_tier_hit_rate"]) else 0.0
    major_jump_hit = tail_metrics["major_jump_hit_rate"] if pd.notna(tail_metrics["major_jump_hit_rate"]) else 0.0
    role_hit = tail_metrics["role_expansion_hit_rate"] if pd.notna(tail_metrics["role_expansion_hit_rate"]) else 0.0
    breakout_avg_rank = (
        tail_metrics["actual_breakout_ceiling_avg_rank"]
        if pd.notna(tail_metrics["actual_breakout_ceiling_avg_rank"])
        else float(get_top_pool_size(scope))
    )
    elite_precision = tail_metrics["elite_precision"] if pd.notna(tail_metrics["elite_precision"]) else 0.0
    elite_recall = tail_metrics["elite_recall"] if pd.notna(tail_metrics["elite_recall"]) else 0.0
    return float(
        (11.0 * (1.0 - top_tier_hit))
        + (8.0 * (1.0 - major_jump_hit))
        + (5.0 * (1.0 - role_hit))
        + (8.0 * (1.0 - max(ceiling_rank_corr, 0.0)))
        + (0.025 * breakout_avg_rank)
        + (3.0 * (1.0 - elite_precision))
        + (3.0 * (1.0 - elite_recall))
    )


def get_calibration_shape_limits(scope: str, mode: str) -> Tuple[float, float]:
    if mode == "ceiling":
        return (
            CEILING_CALIBRATION_MIN_STD_RATIO.get(scope, CEILING_CALIBRATION_MIN_STD_RATIO["ALL"]),
            CEILING_CALIBRATION_MIN_TOP_UNIQUE.get(scope, CEILING_CALIBRATION_MIN_TOP_UNIQUE["ALL"]),
        )
    return (
        POINT_CALIBRATION_MIN_STD_RATIO.get(scope, POINT_CALIBRATION_MIN_STD_RATIO["ALL"]),
        POINT_CALIBRATION_MIN_TOP_UNIQUE.get(scope, POINT_CALIBRATION_MIN_TOP_UNIQUE["ALL"]),
    )


def get_calibration_blend(scope: str, mode: str, method: str) -> float:
    if method == "identity":
        return 1.0
    base_blend_map = CEILING_CALIBRATION_BLEND if mode == "ceiling" else POINT_CALIBRATION_BLEND
    base_blend = float(base_blend_map.get(scope, base_blend_map["ALL"]))
    if method == "linear":
        return min(base_blend + 0.08, 0.96)
    return base_blend


def compute_calibration_shape_metrics(
    raw_values: pd.Series | np.ndarray,
    calibrated_values: pd.Series | np.ndarray,
    scope: str,
) -> Dict[str, float]:
    raw = np.asarray(raw_values, dtype=float)
    calibrated = np.asarray(calibrated_values, dtype=float)
    mask = np.isfinite(raw) & np.isfinite(calibrated)
    if mask.sum() == 0:
        return {
            "std_ratio": 1.0,
            "raw_top_unique_ratio": 1.0,
            "calibrated_top_unique_ratio": 1.0,
        }

    raw = raw[mask]
    calibrated = calibrated[mask]
    raw_std = float(np.std(raw, ddof=0))
    calibrated_std = float(np.std(calibrated, ddof=0))
    std_ratio = calibrated_std / raw_std if raw_std > 1e-9 else 1.0

    pool_size = min(get_top_pool_size(scope), raw.size)
    if pool_size <= 1:
        return {
            "std_ratio": std_ratio,
            "raw_top_unique_ratio": 1.0,
            "calibrated_top_unique_ratio": 1.0,
        }

    raw_top = np.sort(raw)[::-1][:pool_size]
    calibrated_top = np.sort(calibrated)[::-1][:pool_size]
    raw_top_unique_ratio = np.unique(np.round(raw_top, 6)).size / pool_size
    calibrated_top_unique_ratio = np.unique(np.round(calibrated_top, 6)).size / pool_size
    return {
        "std_ratio": std_ratio,
        "raw_top_unique_ratio": float(raw_top_unique_ratio),
        "calibrated_top_unique_ratio": float(calibrated_top_unique_ratio),
    }


def fit_projection_calibration_bundle(
    valid_df: pd.DataFrame,
    predicted_col: str,
    actual_col: str,
    scope: str,
    mode: str = "points",
) -> dict:
    seasons = sorted(valid_df["season"].dropna().unique().tolist())
    if valid_df.empty or len(seasons) < 2:
        return {"method": "identity"}

    calibration_train = valid_df[valid_df["season"] == seasons[0]].copy()
    calibration_test = valid_df[valid_df["season"] == seasons[-1]].copy()
    if calibration_train.empty or calibration_test.empty:
        return {"method": "identity"}

    train_weights = get_calibration_weights(calibration_train, scope, actual_col)
    full_weights = get_calibration_weights(valid_df, scope, actual_col)

    identity_bundle = {"method": "identity"}
    linear_slope, linear_intercept = fit_linear_calibration(
        calibration_train[predicted_col],
        calibration_train[actual_col],
        train_weights,
    )
    linear_bundle = {
        "method": "linear",
        "slope": linear_slope,
        "intercept": linear_intercept,
        "blend": get_calibration_blend(scope, mode, "linear"),
    }
    isotonic_model = fit_regression_isotonic_model(
        calibration_train[predicted_col],
        calibration_train[actual_col],
        train_weights,
    )
    candidate_bundles: Dict[str, dict] = {
        "identity": identity_bundle,
        "linear": linear_bundle,
    }
    if isotonic_model is not None:
        candidate_bundles["isotonic"] = {
            "method": "isotonic",
            "model": isotonic_model,
            "blend": get_calibration_blend(scope, mode, "isotonic"),
        }

    raw_test_values = calibration_test[predicted_col].to_numpy()
    min_std_ratio, min_top_unique_ratio = get_calibration_shape_limits(scope, mode)

    def candidate_loss(bundle: dict) -> float:
        candidate_eval = calibration_test.copy()
        candidate_eval[predicted_col] = apply_regression_calibration_bundle(raw_test_values, bundle)
        shape_metrics = compute_calibration_shape_metrics(raw_test_values, candidate_eval[predicted_col].to_numpy(), scope)
        compression_penalty = 0.0
        if shape_metrics["std_ratio"] < min_std_ratio:
            compression_penalty += 35.0 * (min_std_ratio - shape_metrics["std_ratio"])
        if shape_metrics["calibrated_top_unique_ratio"] < min_top_unique_ratio:
            compression_penalty += 40.0 * (min_top_unique_ratio - shape_metrics["calibrated_top_unique_ratio"])
        unique_ratio_drop = shape_metrics["raw_top_unique_ratio"] - shape_metrics["calibrated_top_unique_ratio"]
        if unique_ratio_drop > CALIBRATION_MAX_UNIQUE_DROP:
            compression_penalty += 30.0 * (unique_ratio_drop - CALIBRATION_MAX_UNIQUE_DROP)
        if mode == "points":
            point_metrics = evaluate_predictions(
                candidate_eval[actual_col],
                candidate_eval[predicted_col].to_numpy(),
            )
            pool_metrics = evaluate_draft_pool(candidate_eval, predicted_col, actual_col, scope)
            return compute_model_loss(point_metrics, pool_metrics) + compression_penalty
        if predicted_col != "predicted_ceiling_points":
            candidate_eval["predicted_ceiling_points"] = np.maximum(
                candidate_eval[predicted_col].to_numpy(),
                candidate_eval.get("predicted_points", candidate_eval[predicted_col]).to_numpy(),
            )
        tail_metrics = evaluate_tail_metrics(candidate_eval, scope)
        return compute_tail_calibration_loss(tail_metrics, scope) + compression_penalty

    candidate_losses = {name: candidate_loss(bundle) for name, bundle in candidate_bundles.items()}
    best_name = min(candidate_losses, key=candidate_losses.get)
    baseline_loss = candidate_losses["identity"]
    best_loss = candidate_losses[best_name]
    if best_name == "identity" or not np.isfinite(best_loss) or best_loss > (baseline_loss - 0.05):
        return identity_bundle

    if best_name == "linear":
        full_slope, full_intercept = fit_linear_calibration(
            valid_df[predicted_col],
            valid_df[actual_col],
            full_weights,
        )
        return {
            "method": "linear",
            "slope": full_slope,
            "intercept": full_intercept,
            "blend": get_calibration_blend(scope, mode, "linear"),
        }

    full_isotonic = fit_regression_isotonic_model(
        valid_df[predicted_col],
        valid_df[actual_col],
        full_weights,
    )
    if full_isotonic is None:
        return identity_bundle
    return {
        "method": "isotonic",
        "model": full_isotonic,
        "blend": get_calibration_blend(scope, mode, "isotonic"),
    }


def build_validation_frame(
    base_df: pd.DataFrame,
    predicted_games: np.ndarray,
    predicted_ppg: np.ndarray,
    adjustment_columns: Dict[str, np.ndarray],
) -> pd.DataFrame:
    validation_frame = base_df.copy()
    validation_frame["predicted_games"] = predicted_games
    validation_frame["predicted_ppg"] = predicted_ppg
    validation_frame["predicted_points_uncalibrated"] = validation_frame["predicted_games"] * validation_frame["predicted_ppg"]
    validation_frame["predicted_vor_uncalibrated"] = (
        validation_frame["predicted_points_uncalibrated"] - validation_frame["replacement_points"]
    )
    for column, values in adjustment_columns.items():
        validation_frame[column] = values
    return validation_frame


def build_availability_downside_target(position_df: pd.DataFrame) -> pd.Series:
    downside_threshold = AVAILABILITY_THRESHOLDS[AVAILABILITY_DOWNSIDE_INDEX - 1]
    return position_df["games_played"].fillna(0.0).lt(downside_threshold).astype(int)


def fit_event_bundle(
    position_train_df: pd.DataFrame,
    position_valid_df: pd.DataFrame,
    feature_columns: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    train_weights: np.ndarray,
    target_builder,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    train_target = target_builder(position_train_df)
    valid_target = target_builder(position_valid_df)
    candidate_models = fit_probability_candidate_models(
        position_train_df,
        feature_columns,
        numeric_features,
        categorical_features,
        train_target,
        train_weights,
    )
    candidate_brier: Dict[str, float] = {}
    candidate_predictions: Dict[str, np.ndarray] = {}
    for model_name, model in candidate_models.items():
        valid_probabilities = predict_binary_probability(model, position_valid_df[feature_columns])
        candidate_predictions[model_name] = valid_probabilities
        candidate_brier[model_name] = brier_score_loss(valid_target, valid_probabilities)
    stack_weights = compute_softmax_weights(candidate_brier, higher_is_better=False)
    valid_raw = weighted_average_series(candidate_predictions, stack_weights)
    calibration_bundle = fit_grouped_probability_calibrator(position_valid_df, valid_raw, valid_target)
    valid_calibrated = apply_grouped_probability_calibrator(valid_raw, position_valid_df, calibration_bundle)
    return {
        "candidate_models": candidate_models,
        "stack_weights": stack_weights,
        "calibration_bundle": calibration_bundle,
        "target_builder": target_builder,
    }, valid_raw, valid_calibrated


def predict_event_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    candidate_predictions = {
        model_name: predict_binary_probability(model, X)
        for model_name, model in bundle["candidate_models"].items()
    }
    raw_probabilities = weighted_average_series(candidate_predictions, bundle["stack_weights"])
    return apply_grouped_probability_calibrator(raw_probabilities, X, bundle["calibration_bundle"])


def fit_point_distribution_table(validation_df: pd.DataFrame) -> dict:
    residuals = validation_df["fantasy_points_total"] - validation_df["predicted_points"]
    position_quantiles = {
        quantile: float(residuals.quantile(quantile)) if not residuals.empty else 0.0
        for quantile in POINT_DISTRIBUTION_QUANTILES
    }
    grouped = (
        validation_df.assign(point_residual=residuals)
        .groupby(["position_archetype", "history_bucket"], dropna=False)
        .agg(group_count=("player_key", "size"))
        .reset_index()
    )
    for quantile in POINT_DISTRIBUTION_QUANTILES:
        quantile_frame = (
            validation_df.assign(point_residual=residuals)
            .groupby(["position_archetype", "history_bucket"], dropna=False)["point_residual"]
            .quantile(quantile)
            .reset_index()
            .rename(columns={"point_residual": f"residual_q{int(quantile * 100)}"})
        )
        grouped = grouped.merge(quantile_frame, on=["position_archetype", "history_bucket"], how="left")
    return {
        "position_quantiles": position_quantiles,
        "group_table": grouped,
    }


def apply_point_distribution(target_df: pd.DataFrame, predicted_points: pd.Series, distribution_bundle: dict) -> Dict[str, np.ndarray]:
    if not distribution_bundle:
        return {
            f"predicted_points_p{int(quantile * 100)}": predicted_points.to_numpy()
            for quantile in POINT_DISTRIBUTION_QUANTILES
        }

    group_table = distribution_bundle.get("group_table", pd.DataFrame())
    position_quantiles = distribution_bundle.get("position_quantiles", {})
    if isinstance(group_table, pd.DataFrame) and not group_table.empty:
        merge_df = (
            target_df[["position_archetype", "history_bucket"]]
            .reset_index()
            .merge(group_table, on=["position_archetype", "history_bucket"], how="left")
            .set_index("index")
            .reindex(target_df.index)
        )
    else:
        merge_df = pd.DataFrame(index=target_df.index)

    group_count = merge_df.get("group_count", pd.Series(0.0, index=target_df.index)).fillna(0.0)
    shrink = (group_count / (group_count + MIXTURE_EXPERT_SHRINKAGE)).clip(0.0, 1.0)
    quantile_columns: Dict[str, np.ndarray] = {}
    monotonic_stack: List[np.ndarray] = []
    for quantile in POINT_DISTRIBUTION_QUANTILES:
        column = f"residual_q{int(quantile * 100)}"
        base_quantile = float(position_quantiles.get(quantile, 0.0))
        group_quantile = merge_df.get(column, pd.Series(base_quantile, index=target_df.index)).fillna(base_quantile)
        residual_quantile = (shrink * group_quantile) + ((1.0 - shrink) * base_quantile)
        quantile_prediction = np.clip(predicted_points + residual_quantile, 0.0, None).to_numpy()
        monotonic_stack.append(quantile_prediction)
        quantile_columns[f"predicted_points_p{int(quantile * 100)}"] = quantile_prediction

    monotonic_matrix = np.maximum.accumulate(np.column_stack(monotonic_stack), axis=1)
    for idx, quantile in enumerate(POINT_DISTRIBUTION_QUANTILES):
        quantile_columns[f"predicted_points_p{int(quantile * 100)}"] = monotonic_matrix[:, idx]
    return quantile_columns


def fit_archetype_expert_adjustments(validation_df: pd.DataFrame) -> dict:
    residuals = validation_df["fantasy_points_total"] - validation_df["predicted_points"]
    fallback = float(residuals.mean()) if not residuals.empty else 0.0
    grouped = (
        validation_df.assign(point_residual=residuals)
        .groupby(["position_archetype", "history_bucket"], dropna=False)["point_residual"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "group_mean_residual", "size": "group_count"})
    )
    return {"fallback": fallback, "group_table": grouped}


def apply_archetype_expert_adjustments(target_df: pd.DataFrame, expert_bundle: dict) -> np.ndarray:
    if not expert_bundle:
        return np.zeros(len(target_df), dtype=float)

    group_table = expert_bundle.get("group_table", pd.DataFrame())
    if isinstance(group_table, pd.DataFrame) and not group_table.empty:
        merge_df = (
            target_df[["position_archetype", "history_bucket"]]
            .reset_index()
            .merge(group_table, on=["position_archetype", "history_bucket"], how="left")
            .set_index("index")
            .reindex(target_df.index)
        )
    else:
        merge_df = pd.DataFrame(index=target_df.index)

    group_mean = merge_df.get("group_mean_residual", pd.Series(expert_bundle.get("fallback", 0.0), index=target_df.index))
    group_count = merge_df.get("group_count", pd.Series(0.0, index=target_df.index)).fillna(0.0)
    shrink = (group_count / (group_count + MIXTURE_EXPERT_SHRINKAGE)).clip(0.0, 1.0)
    return ((shrink * group_mean.fillna(expert_bundle.get("fallback", 0.0))) + ((1.0 - shrink) * expert_bundle.get("fallback", 0.0))).to_numpy()


def fit_upside_archetype_routing(validation_df: pd.DataFrame) -> dict:
    if validation_df.empty:
        return {}

    working = validation_df.copy()
    ceiling_anchor_col = "predicted_ceiling_points_direct" if "predicted_ceiling_points_direct" in working.columns else "predicted_points"
    working["ceiling_points_target"] = build_ceiling_points_target(working)
    working["actual_top_tier"] = build_top_tier_finish_target(working)
    working["actual_major_jump"] = build_major_jump_target(working)
    working["actual_role_expansion"] = build_role_expansion_target(working)
    working["ceiling_adjustment"] = working["ceiling_points_target"] - working[ceiling_anchor_col].fillna(working["predicted_points"])
    working["top_tier_adjustment"] = working["actual_top_tier"] - working["top_tier_finish_probability"].fillna(0.0)
    working["major_jump_adjustment"] = working["actual_major_jump"] - working["major_jump_probability"].fillna(0.0)
    working["role_expansion_adjustment"] = working["actual_role_expansion"] - working["role_expansion_probability"].fillna(0.0)

    grouped = (
        working.groupby(["position_archetype", "history_bucket"], dropna=False)
        .agg(
            group_count=("player_key", "size"),
            ceiling_adjustment_mean=("ceiling_adjustment", "mean"),
            top_tier_adjustment_mean=("top_tier_adjustment", "mean"),
            major_jump_adjustment_mean=("major_jump_adjustment", "mean"),
            role_expansion_adjustment_mean=("role_expansion_adjustment", "mean"),
        )
        .reset_index()
    )
    return {
        "fallbacks": {
            "ceiling_adjustment_mean": float(working["ceiling_adjustment"].mean()) if not working.empty else 0.0,
            "top_tier_adjustment_mean": float(working["top_tier_adjustment"].mean()) if not working.empty else 0.0,
            "major_jump_adjustment_mean": float(working["major_jump_adjustment"].mean()) if not working.empty else 0.0,
            "role_expansion_adjustment_mean": float(working["role_expansion_adjustment"].mean()) if not working.empty else 0.0,
        },
        "group_table": grouped,
    }


def apply_upside_archetype_routing(target_df: pd.DataFrame, routing_bundle: dict) -> Dict[str, np.ndarray]:
    if not routing_bundle:
        zero = np.zeros(len(target_df), dtype=float)
        return {
            "ceiling_points_adjustment": zero,
            "top_tier_probability_adjustment": zero,
            "major_jump_probability_adjustment": zero,
            "role_expansion_probability_adjustment": zero,
        }

    group_table = routing_bundle.get("group_table", pd.DataFrame())
    fallbacks = routing_bundle.get("fallbacks", {})
    if isinstance(group_table, pd.DataFrame) and not group_table.empty:
        merge_df = (
            target_df[["position_archetype", "history_bucket"]]
            .reset_index()
            .merge(group_table, on=["position_archetype", "history_bucket"], how="left")
            .set_index("index")
            .reindex(target_df.index)
        )
    else:
        merge_df = pd.DataFrame(index=target_df.index)

    group_count = merge_df.get("group_count", pd.Series(0.0, index=target_df.index)).fillna(0.0)
    shrink = (group_count / (group_count + MIXTURE_EXPERT_SHRINKAGE)).clip(0.0, 1.0)

    def blended_adjustment(column: str, clip_low: float, clip_high: float, scale: float) -> np.ndarray:
        fallback = float(fallbacks.get(column, 0.0))
        group_values = merge_df.get(column, pd.Series(fallback, index=target_df.index)).fillna(fallback)
        values = ((shrink * group_values) + ((1.0 - shrink) * fallback)).to_numpy(dtype=float)
        return np.clip(scale * values, clip_low, clip_high)

    return {
        "ceiling_points_adjustment": blended_adjustment("ceiling_adjustment_mean", -35.0, 45.0, 0.55),
        "top_tier_probability_adjustment": blended_adjustment("top_tier_adjustment_mean", -0.18, 0.18, 0.65),
        "major_jump_probability_adjustment": blended_adjustment("major_jump_adjustment_mean", -0.20, 0.20, 0.70),
        "role_expansion_probability_adjustment": blended_adjustment("role_expansion_adjustment_mean", -0.20, 0.20, 0.70),
    }


def train_model(feature_df: pd.DataFrame) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    numeric_features, categorical_features = get_model_features(feature_df)
    model_features = numeric_features + categorical_features

    train_mask = (feature_df["season"] >= MIN_MODEL_SEASON) & (feature_df["season"] < 2022)
    valid_mask = feature_df["season"].isin([2022, 2023])
    final_train_mask = (feature_df["season"] >= MIN_MODEL_SEASON) & (feature_df["season"] < 2024)

    train_df = feature_df.loc[train_mask].copy()
    valid_df = feature_df.loc[valid_mask].copy()
    final_train_df = feature_df.loc[final_train_mask].copy()
    target_2024_df = feature_df.loc[feature_df["season"] == 2024].copy()
    validation_priors = build_projection_priors(train_df)
    final_priors = build_projection_priors(final_train_df)

    comparison_rows: List[dict] = []
    validation_outputs: List[pd.DataFrame] = []
    position_models: Dict[str, dict] = {}

    for position in REPLACEMENT_RANKS:
        position_train_df = train_df[train_df["position"] == position].copy()
        position_valid_df = valid_df[valid_df["position"] == position].copy()
        position_final_train_df = final_train_df[final_train_df["position"] == position].copy()
        position_target_2024_df = target_2024_df[target_2024_df["position"] == position].copy()

        train_weights = compute_recency_weights(position_train_df) * compute_density_ratio_weights(
            position_train_df,
            position_target_2024_df,
            model_features,
            numeric_features,
            categorical_features,
        )
        final_train_weights = compute_recency_weights(position_final_train_df) * compute_density_ratio_weights(
            position_final_train_df,
            position_target_2024_df,
            model_features,
            numeric_features,
            categorical_features,
        )

        availability_candidates = fit_availability_candidate_model_set(
            position_train_df,
            model_features,
            numeric_features,
            categorical_features,
            train_weights,
        )
        ppg_candidates = fit_candidate_model_set(
            position_train_df,
            model_features,
            "fantasy_points_per_game",
            numeric_features,
            categorical_features,
            train_weights,
        )

        availability_predictions = {
            model_name: predict_availability_distribution(bundle, position_valid_df[model_features])
            for model_name, bundle in availability_candidates.items()
        }
        ppg_predictions = {
            model_name: model.predict(position_valid_df[model_features]) for model_name, model in ppg_candidates.items()
        }

        combo_outputs: Dict[str, dict] = {}
        combo_losses: Dict[str, float] = {}
        combo_calibration_info: Dict[str, dict] = {}

        for availability_model_name, availability_tuple in availability_predictions.items():
            valid_games_raw, valid_availability_downside, valid_bucket_probabilities = availability_tuple
            for ppg_model_name, valid_ppg_raw in ppg_predictions.items():
                combo_name = f"{availability_model_name}+{ppg_model_name}"
                valid_games_adj, valid_ppg_adj, adjustment_columns = apply_projection_priors(
                    position_valid_df,
                    predicted_games=valid_games_raw,
                    predicted_ppg=valid_ppg_raw,
                    projection_priors=validation_priors,
                )
                position_validation = build_validation_frame(
                    position_valid_df,
                    predicted_games=valid_games_adj,
                    predicted_ppg=valid_ppg_adj,
                    adjustment_columns=adjustment_columns,
                )
                slope, intercept, use_calibration = decide_tail_calibration(position_validation)
                if use_calibration:
                    calibration_weights = get_calibration_weights(position_validation, position, "fantasy_points_total")
                    slope, intercept = fit_linear_calibration(
                        position_validation["predicted_points_uncalibrated"],
                        position_validation["fantasy_points_total"],
                        calibration_weights,
                    )
                else:
                    slope, intercept = 1.0, 0.0

                position_validation["availability_downside_probability"] = valid_availability_downside
                position_validation["predicted_points"] = (
                    intercept + (slope * position_validation["predicted_points_uncalibrated"])
                ).clip(lower=0.0)
                position_validation["predicted_vor"] = (
                    position_validation["predicted_points"] - position_validation["replacement_points"]
                )
                position_validation["selected_model"] = combo_name

                point_metrics = evaluate_predictions(
                    position_validation["fantasy_points_total"],
                    position_validation["predicted_points"].to_numpy(),
                )
                vor_metrics = evaluate_predictions(
                    position_validation["vor"],
                    position_validation["predicted_vor"].to_numpy(),
                )
                pool_metrics = evaluate_draft_pool(
                    position_validation,
                    predicted_col="predicted_points",
                    actual_col="fantasy_points_total",
                    scope=position,
                )
                comparison_rows.append(
                    {
                        "scope": position,
                        "model": combo_name,
                        "point_rmse": point_metrics["rmse"],
                        "point_mae": point_metrics["mae"],
                        "point_corr": point_metrics["corr"],
                        "vor_rmse": vor_metrics["rmse"],
                        "vor_mae": vor_metrics["mae"],
                        "vor_corr": vor_metrics["corr"],
                        **pool_metrics,
                        **get_nan_tail_metrics(),
                    }
                )
                combo_outputs[combo_name] = {
                    "frame": position_validation,
                    "availability_model_name": availability_model_name,
                    "ppg_model_name": ppg_model_name,
                    "bucket_probabilities": valid_bucket_probabilities,
                }
                combo_losses[combo_name] = compute_model_loss(point_metrics, pool_metrics)
                combo_calibration_info[combo_name] = {
                    "calibration_slope": slope,
                    "calibration_intercept": intercept,
                    "used_calibration": use_calibration,
                }

        combo_weights = compute_softmax_weights(combo_losses, higher_is_better=False)
        combo_summary = ", ".join(f"{name}:{weight:.2f}" for name, weight in sorted(combo_weights.items()))

        def combine_combo_predictions(source_df: pd.DataFrame, availability_map: Dict[str, dict], ppg_map: Dict[str, Pipeline], priors: Dict[str, object]) -> Tuple[pd.DataFrame, np.ndarray]:
            output_frames: Dict[str, pd.DataFrame] = {}
            bucket_probabilities: Dict[str, np.ndarray] = {}
            for combo_name, combo_meta in combo_outputs.items():
                games_raw, downside_raw, bucket_probs = predict_availability_distribution(
                    availability_map[combo_meta["availability_model_name"]],
                    source_df[model_features],
                )
                ppg_raw = ppg_map[combo_meta["ppg_model_name"]].predict(source_df[model_features])
                games_adj, ppg_adj, adjustment_columns = apply_projection_priors(
                    source_df,
                    predicted_games=games_raw,
                    predicted_ppg=ppg_raw,
                    projection_priors=priors,
                )
                combo_frame = build_validation_frame(
                    source_df,
                    predicted_games=games_adj,
                    predicted_ppg=ppg_adj,
                    adjustment_columns=adjustment_columns,
                )
                combo_frame["availability_downside_probability"] = downside_raw
                calibration_meta = combo_calibration_info[combo_name]
                combo_frame["predicted_points"] = (
                    calibration_meta["calibration_intercept"]
                    + (calibration_meta["calibration_slope"] * combo_frame["predicted_points_uncalibrated"])
                ).clip(lower=0.0)
                combo_frame["predicted_vor"] = combo_frame["predicted_points"] - combo_frame["replacement_points"]
                combo_frame["selected_model"] = combo_name
                output_frames[combo_name] = combo_frame
                bucket_probabilities[combo_name] = bucket_probs

            combined = source_df.copy()
            combined["predicted_games"] = weighted_average_series(
                {name: frame["predicted_games"].to_numpy() for name, frame in output_frames.items()},
                combo_weights,
            )
            combined["predicted_ppg"] = weighted_average_series(
                {name: frame["predicted_ppg"].to_numpy() for name, frame in output_frames.items()},
                combo_weights,
            )
            combined["predicted_points_uncalibrated"] = weighted_average_series(
                {name: frame["predicted_points_uncalibrated"].to_numpy() for name, frame in output_frames.items()},
                combo_weights,
            )
            combined["predicted_points"] = weighted_average_series(
                {name: frame["predicted_points"].to_numpy() for name, frame in output_frames.items()},
                combo_weights,
            )
            combined["predicted_vor"] = combined["predicted_points"] - combined["replacement_points"]
            combined["availability_downside_probability"] = weighted_average_series(
                {name: frame["availability_downside_probability"].to_numpy() for name, frame in output_frames.items()},
                combo_weights,
            )
            for adjustment_column in [
                "rookie_games_adjustment",
                "rookie_ppg_adjustment",
                "comeback_games_adjustment",
                "comeback_ppg_adjustment",
                "thin_history_games_adjustment",
                "thin_history_ppg_adjustment",
            ]:
                combined[adjustment_column] = weighted_average_series(
                    {name: frame[adjustment_column].to_numpy() for name, frame in output_frames.items()},
                    combo_weights,
                )
            combined_bucket_probabilities = weighted_average_series(bucket_probabilities, combo_weights)
            combined["selected_model"] = f"Stack[{combo_summary}]"
            return combined, combined_bucket_probabilities

        combined_validation, valid_bucket_probabilities = combine_combo_predictions(
            position_valid_df,
            availability_candidates,
            ppg_candidates,
            validation_priors,
        )

        availability_calibration_bundle = fit_grouped_probability_calibrator(
            position_valid_df,
            combined_validation["availability_downside_probability"].to_numpy(),
            build_availability_downside_target(position_valid_df),
        )
        combined_validation["availability_downside_probability"] = apply_grouped_probability_calibrator(
            combined_validation["availability_downside_probability"].to_numpy(),
            position_valid_df,
            availability_calibration_bundle,
        )

        train_combined_frame, _ = combine_combo_predictions(
            position_train_df,
            availability_candidates,
            ppg_candidates,
            validation_priors,
        )
        train_combined_frame["availability_downside_probability"] = apply_grouped_probability_calibrator(
            train_combined_frame["availability_downside_probability"].to_numpy(),
            position_train_df,
            availability_calibration_bundle,
        )

        train_top_tier_target = build_top_tier_finish_target(position_train_df)
        train_major_jump_target = build_major_jump_target(position_train_df)
        train_role_expansion_target = build_role_expansion_target(position_train_df)

        elite_bundle, _, elite_valid_prob = fit_event_bundle(
            position_train_df,
            position_valid_df,
            model_features,
            numeric_features,
            categorical_features,
            train_weights,
            build_elite_finish_target,
        )
        top_tier_bundle, _, top_tier_valid_prob = fit_event_bundle(
            position_train_df,
            position_valid_df,
            model_features,
            numeric_features,
            categorical_features,
            train_weights,
            build_top_tier_finish_target,
        )
        major_jump_bundle, _, major_jump_valid_prob = fit_event_bundle(
            position_train_df,
            position_valid_df,
            model_features,
            numeric_features,
            categorical_features,
            train_weights,
            build_major_jump_target,
        )
        role_expansion_bundle, _, role_expansion_valid_prob = fit_event_bundle(
            position_train_df,
            position_valid_df,
            model_features,
            numeric_features,
            categorical_features,
            train_weights,
            build_role_expansion_target,
        )
        collapse_bundle, _, collapse_valid_prob = fit_event_bundle(
            position_train_df,
            position_valid_df,
            model_features,
            numeric_features,
            categorical_features,
            train_weights,
            build_collapse_target,
        )

        combined_validation["elite_finish_probability"] = elite_valid_prob
        combined_validation["top_tier_finish_probability"] = top_tier_valid_prob
        combined_validation["major_jump_probability"] = major_jump_valid_prob
        combined_validation["role_expansion_probability"] = role_expansion_valid_prob
        combined_validation["breakout_probability"] = np.clip(
            (0.30 * combined_validation["top_tier_finish_probability"])
            + (0.45 * combined_validation["major_jump_probability"])
            + (0.25 * combined_validation["role_expansion_probability"]),
            0.0,
            1.0,
        )
        combined_validation["collapse_probability"] = collapse_valid_prob
        train_combined_frame["elite_finish_probability"] = predict_event_bundle(elite_bundle, position_train_df[model_features])
        train_combined_frame["top_tier_finish_probability"] = predict_event_bundle(
            top_tier_bundle,
            position_train_df[model_features],
        )
        train_combined_frame["major_jump_probability"] = predict_event_bundle(
            major_jump_bundle,
            position_train_df[model_features],
        )
        train_combined_frame["role_expansion_probability"] = predict_event_bundle(
            role_expansion_bundle,
            position_train_df[model_features],
        )
        train_combined_frame["breakout_probability"] = np.clip(
            (0.30 * train_combined_frame["top_tier_finish_probability"])
            + (0.45 * train_combined_frame["major_jump_probability"])
            + (0.25 * train_combined_frame["role_expansion_probability"]),
            0.0,
            1.0,
        )
        train_combined_frame["collapse_probability"] = predict_event_bundle(collapse_bundle, position_train_df[model_features])

        ceiling_numeric_features = numeric_features + [
            "predicted_games",
            "predicted_ppg",
            "predicted_points",
            "predicted_vor",
            "elite_finish_probability",
            "top_tier_finish_probability",
            "major_jump_probability",
            "role_expansion_probability",
            "breakout_probability",
            "collapse_probability",
            "availability_downside_probability",
        ]
        ceiling_feature_columns = ceiling_numeric_features + categorical_features
        ceiling_train_features = build_ceiling_feature_frame(train_combined_frame, model_features)
        ceiling_valid_features = build_ceiling_feature_frame(combined_validation, model_features)
        ceiling_train_frame = ceiling_train_features.copy()
        ceiling_valid_target = build_ceiling_points_target(position_valid_df)
        ceiling_train_frame["ceiling_points_target"] = build_ceiling_points_target(position_train_df)
        ceiling_train_weights = train_weights * (
            1.0
            + (5.0 * train_top_tier_target.to_numpy(dtype=float))
            + (4.0 * train_major_jump_target.to_numpy(dtype=float))
            + (2.5 * train_role_expansion_target.to_numpy(dtype=float))
        )
        ceiling_candidates = fit_candidate_model_set(
            ceiling_train_frame,
            ceiling_feature_columns,
            "ceiling_points_target",
            ceiling_numeric_features,
            categorical_features,
            ceiling_train_weights,
        )
        ceiling_candidate_predictions: Dict[str, np.ndarray] = {}
        ceiling_candidate_losses: Dict[str, float] = {}
        for ceiling_name, ceiling_model in ceiling_candidates.items():
            candidate_prediction = np.clip(
                ceiling_model.predict(ceiling_valid_features[ceiling_feature_columns]),
                combined_validation["predicted_points"].to_numpy(),
                None,
            )
            candidate_frame = combined_validation.copy()
            candidate_frame["predicted_ceiling_points_direct"] = candidate_prediction
            candidate_frame["predicted_ceiling_points"] = np.maximum(
                candidate_prediction,
                candidate_frame["predicted_points"].to_numpy(),
            )
            candidate_tail_metrics = evaluate_tail_metrics(candidate_frame, position)
            ceiling_point_metrics = evaluate_predictions(ceiling_valid_target, candidate_prediction)
            ceiling_rank_corr = (
                candidate_tail_metrics["ceiling_top_pool_rank_corr"]
                if pd.notna(candidate_tail_metrics["ceiling_top_pool_rank_corr"])
                else 0.0
            )
            top_tier_hit = candidate_tail_metrics["top_tier_hit_rate"] if pd.notna(candidate_tail_metrics["top_tier_hit_rate"]) else 0.0
            major_jump_hit = candidate_tail_metrics["major_jump_hit_rate"] if pd.notna(candidate_tail_metrics["major_jump_hit_rate"]) else 0.0
            role_hit = candidate_tail_metrics["role_expansion_hit_rate"] if pd.notna(candidate_tail_metrics["role_expansion_hit_rate"]) else 0.0
            breakout_avg_rank = (
                candidate_tail_metrics["actual_breakout_ceiling_avg_rank"]
                if pd.notna(candidate_tail_metrics["actual_breakout_ceiling_avg_rank"])
                else float(get_top_pool_size(position))
            )
            ceiling_candidate_predictions[ceiling_name] = candidate_prediction
            ceiling_candidate_losses[ceiling_name] = float(
                (0.15 * ceiling_point_metrics["mae"])
                + (12.0 * (1.0 - top_tier_hit))
                + (8.0 * (1.0 - major_jump_hit))
                + (6.0 * (1.0 - role_hit))
                + (8.0 * (1.0 - max(ceiling_rank_corr, 0.0)))
                + (0.03 * breakout_avg_rank)
            )

        ceiling_weights = compute_softmax_weights(ceiling_candidate_losses, higher_is_better=False)
        combined_validation["predicted_ceiling_points_direct"] = weighted_average_series(
            ceiling_candidate_predictions,
            ceiling_weights,
        )
        combined_validation["predicted_ceiling_points"] = np.maximum(
            combined_validation["predicted_ceiling_points_direct"].to_numpy(),
            combined_validation["predicted_points"].to_numpy(),
        )
        train_ceiling_predictions = {
            name: np.clip(
                model.predict(ceiling_train_features[ceiling_feature_columns]),
                train_combined_frame["predicted_points"].to_numpy(),
                None,
            )
            for name, model in ceiling_candidates.items()
        }
        train_combined_frame["predicted_ceiling_points_direct"] = weighted_average_series(
            train_ceiling_predictions,
            ceiling_weights,
        )
        train_combined_frame["predicted_ceiling_points"] = np.maximum(
            train_combined_frame["predicted_ceiling_points_direct"].to_numpy(),
            train_combined_frame["predicted_points"].to_numpy(),
        )

        upside_archetype_bundle = fit_upside_archetype_routing(combined_validation)
        valid_upside_adjustments = apply_upside_archetype_routing(combined_validation, upside_archetype_bundle)
        combined_validation["upside_archetype_ceiling_adjustment"] = valid_upside_adjustments["ceiling_points_adjustment"]
        combined_validation["predicted_ceiling_points_direct"] = np.maximum(
            combined_validation["predicted_points"].to_numpy(),
            combined_validation["predicted_ceiling_points_direct"].to_numpy()
            + combined_validation["upside_archetype_ceiling_adjustment"].to_numpy(),
        )
        combined_validation["predicted_ceiling_points"] = np.maximum(
            combined_validation["predicted_points"].to_numpy(),
            combined_validation["predicted_ceiling_points"].to_numpy()
            + combined_validation["upside_archetype_ceiling_adjustment"].to_numpy(),
        )
        combined_validation["top_tier_finish_probability"] = np.clip(
            combined_validation["top_tier_finish_probability"].to_numpy()
            + valid_upside_adjustments["top_tier_probability_adjustment"],
            0.0,
            1.0,
        )
        combined_validation["major_jump_probability"] = np.clip(
            combined_validation["major_jump_probability"].to_numpy()
            + valid_upside_adjustments["major_jump_probability_adjustment"],
            0.0,
            1.0,
        )
        combined_validation["role_expansion_probability"] = np.clip(
            combined_validation["role_expansion_probability"].to_numpy()
            + valid_upside_adjustments["role_expansion_probability_adjustment"],
            0.0,
            1.0,
        )
        combined_validation["breakout_probability"] = np.clip(
            (0.30 * combined_validation["top_tier_finish_probability"])
            + (0.45 * combined_validation["major_jump_probability"])
            + (0.25 * combined_validation["role_expansion_probability"]),
            0.0,
            1.0,
        )

        train_upside_adjustments = apply_upside_archetype_routing(train_combined_frame, upside_archetype_bundle)
        train_combined_frame["upside_archetype_ceiling_adjustment"] = train_upside_adjustments["ceiling_points_adjustment"]
        train_combined_frame["predicted_ceiling_points_direct"] = np.maximum(
            train_combined_frame["predicted_points"].to_numpy(),
            train_combined_frame["predicted_ceiling_points_direct"].to_numpy()
            + train_combined_frame["upside_archetype_ceiling_adjustment"].to_numpy(),
        )
        train_combined_frame["predicted_ceiling_points"] = np.maximum(
            train_combined_frame["predicted_points"].to_numpy(),
            train_combined_frame["predicted_ceiling_points"].to_numpy()
            + train_combined_frame["upside_archetype_ceiling_adjustment"].to_numpy(),
        )
        train_combined_frame["top_tier_finish_probability"] = np.clip(
            train_combined_frame["top_tier_finish_probability"].to_numpy()
            + train_upside_adjustments["top_tier_probability_adjustment"],
            0.0,
            1.0,
        )
        train_combined_frame["major_jump_probability"] = np.clip(
            train_combined_frame["major_jump_probability"].to_numpy()
            + train_upside_adjustments["major_jump_probability_adjustment"],
            0.0,
            1.0,
        )
        train_combined_frame["role_expansion_probability"] = np.clip(
            train_combined_frame["role_expansion_probability"].to_numpy()
            + train_upside_adjustments["role_expansion_probability_adjustment"],
            0.0,
            1.0,
        )
        train_combined_frame["breakout_probability"] = np.clip(
            (0.30 * train_combined_frame["top_tier_finish_probability"])
            + (0.45 * train_combined_frame["major_jump_probability"])
            + (0.25 * train_combined_frame["role_expansion_probability"]),
            0.0,
            1.0,
        )

        ranker_numeric_features = numeric_features + [
            "predicted_games",
            "predicted_ppg",
            "predicted_points",
            "predicted_vor",
            "elite_finish_probability",
            "top_tier_finish_probability",
            "major_jump_probability",
            "role_expansion_probability",
            "predicted_ceiling_points",
            "breakout_probability",
            "collapse_probability",
            "availability_downside_probability",
        ]
        ranker_feature_columns = ranker_numeric_features + categorical_features
        ranker_train_features = build_ranker_feature_frame(train_combined_frame, model_features)
        ranker_valid_features = build_ranker_feature_frame(combined_validation, model_features)
        ranker_target = build_ranker_target(position_train_df)
        ranker_weights = train_weights * (1.0 + (12.0 * ranker_target.to_numpy()))
        ranker_bonus_scale = get_ranker_bonus_scale(combined_validation)

        base_point_metrics = evaluate_predictions(
            combined_validation["fantasy_points_total"],
            combined_validation["predicted_points"].to_numpy(),
        )
        base_pool_metrics = evaluate_draft_pool(
            combined_validation,
            predicted_col="predicted_points",
            actual_col="fantasy_points_total",
            scope=position,
        )
        ranker_candidate_predictions: Dict[str, np.ndarray] = {}
        ranker_improvements: Dict[str, float] = {}
        ranker_metric_rows: Dict[str, dict] = {}
        for ranker_name in ["Ridge", "HistGB"]:
            ranker_model = build_model_by_name(ranker_name, ranker_numeric_features, categorical_features)
            ranker_model.fit(
                ranker_train_features[ranker_feature_columns],
                ranker_target,
                model__sample_weight=ranker_weights,
            )
            valid_rank_scores = ranker_model.predict(ranker_valid_features[ranker_feature_columns])
            ranked_validation = combined_validation.copy()
            ranked_validation["draft_ranker_raw"] = valid_rank_scores
            ranked_validation["draft_ranker_bonus"] = ranker_bonus_scale * standardize_array(valid_rank_scores)
            ranked_validation["predicted_points_ranked"] = (
                ranked_validation["predicted_points"] + ranked_validation["draft_ranker_bonus"]
            ).clip(lower=0.0)
            ranked_validation["predicted_vor_ranked"] = (
                ranked_validation["predicted_points_ranked"] - ranked_validation["replacement_points"]
            )
            ranked_point_metrics = evaluate_predictions(
                ranked_validation["fantasy_points_total"],
                ranked_validation["predicted_points_ranked"].to_numpy(),
            )
            ranked_pool_metrics = evaluate_draft_pool(
                ranked_validation,
                predicted_col="predicted_points_ranked",
                actual_col="fantasy_points_total",
                scope=position,
            )
            ranked_rank_corr = (
                ranked_pool_metrics["top_pool_rank_corr"]
                if pd.notna(ranked_pool_metrics["top_pool_rank_corr"])
                else 0.0
            )
            base_rank_corr = (
                base_pool_metrics["top_pool_rank_corr"]
                if pd.notna(base_pool_metrics["top_pool_rank_corr"])
                else 0.0
            )
            improvement_score = (
                max(ranked_pool_metrics["top_pool_hit_rate"] - base_pool_metrics["top_pool_hit_rate"], 0.0)
                + (0.8 * max(ranked_rank_corr - base_rank_corr, 0.0))
                - (0.02 * max(ranked_pool_metrics["top_pool_mae"] - base_pool_metrics["top_pool_mae"], 0.0))
            )
            ranker_candidate_predictions[ranker_name] = valid_rank_scores
            ranker_metric_rows[ranker_name] = {
                "point_metrics": ranked_point_metrics,
                "pool_metrics": ranked_pool_metrics,
            }
            if improvement_score > 0:
                ranker_improvements[ranker_name] = improvement_score

        ranker_weights_map = compute_softmax_weights(ranker_improvements, higher_is_better=True) if ranker_improvements else {}
        ranker_summary = "None"
        if ranker_weights_map:
            combined_rank_scores = weighted_average_series(ranker_candidate_predictions, ranker_weights_map)
            ranked_validation = combined_validation.copy()
            ranked_validation["draft_ranker_raw"] = combined_rank_scores
            ranked_validation["draft_ranker_bonus"] = ranker_bonus_scale * standardize_array(combined_rank_scores)
            ranked_validation["predicted_points_ranked"] = (
                ranked_validation["predicted_points"] + ranked_validation["draft_ranker_bonus"]
            ).clip(lower=0.0)
            ranked_validation["predicted_vor_ranked"] = (
                ranked_validation["predicted_points_ranked"] - ranked_validation["replacement_points"]
            )
            ranked_point_metrics = evaluate_predictions(
                ranked_validation["fantasy_points_total"],
                ranked_validation["predicted_points_ranked"].to_numpy(),
            )
            ranked_pool_metrics = evaluate_draft_pool(
                ranked_validation,
                predicted_col="predicted_points_ranked",
                actual_col="fantasy_points_total",
                scope=position,
            )
            ranked_rank_corr_safe = (
                ranked_pool_metrics["top_pool_rank_corr"]
                if pd.notna(ranked_pool_metrics["top_pool_rank_corr"])
                else 0.0
            )
            base_rank_corr_safe = (
                base_pool_metrics["top_pool_rank_corr"]
                if pd.notna(base_pool_metrics["top_pool_rank_corr"])
                else 0.0
            )
            if (
                ranked_pool_metrics["top_pool_hit_rate"] >= base_pool_metrics["top_pool_hit_rate"]
                and ranked_rank_corr_safe >= base_rank_corr_safe - 0.01
                and ranked_pool_metrics["top_pool_mae"] <= base_pool_metrics["top_pool_mae"] * 1.04
            ):
                combined_validation = ranked_validation
                combined_validation["predicted_points"] = combined_validation["predicted_points_ranked"]
                combined_validation["predicted_vor"] = combined_validation["predicted_vor_ranked"]
                ranker_summary = ", ".join(f"{name}:{weight:.2f}" for name, weight in sorted(ranker_weights_map.items()))
                train_ranker_candidate_predictions: Dict[str, np.ndarray] = {}
                for ranker_name in ranker_weights_map:
                    ranker_model = build_model_by_name(ranker_name, ranker_numeric_features, categorical_features)
                    ranker_model.fit(
                        ranker_train_features[ranker_feature_columns],
                        ranker_target,
                        model__sample_weight=ranker_weights,
                    )
                    train_ranker_candidate_predictions[ranker_name] = ranker_model.predict(
                        ranker_train_features[ranker_feature_columns]
                    )
                train_combined_rank_scores = weighted_average_series(train_ranker_candidate_predictions, ranker_weights_map)
                train_combined_frame["draft_ranker_raw"] = train_combined_rank_scores
                train_combined_frame["draft_ranker_bonus"] = ranker_bonus_scale * standardize_array(train_combined_rank_scores)
                train_combined_frame["predicted_points"] = (
                    train_combined_frame["predicted_points"] + train_combined_frame["draft_ranker_bonus"]
                ).clip(lower=0.0)
                train_combined_frame["predicted_vor"] = train_combined_frame["predicted_points"] - train_combined_frame["replacement_points"]
            else:
                ranker_weights_map = {}

        if not ranker_weights_map:
            combined_validation["draft_ranker_raw"] = 0.0
            combined_validation["draft_ranker_bonus"] = 0.0
            train_combined_frame["draft_ranker_raw"] = 0.0
            train_combined_frame["draft_ranker_bonus"] = 0.0

        archetype_expert_bundle = fit_archetype_expert_adjustments(combined_validation)
        expert_adjustment_valid = 0.35 * apply_archetype_expert_adjustments(combined_validation, archetype_expert_bundle)
        expert_candidate = combined_validation.copy()
        expert_candidate["archetype_expert_adjustment"] = expert_adjustment_valid
        expert_candidate["predicted_points"] = (expert_candidate["predicted_points"] + expert_adjustment_valid).clip(lower=0.0)
        expert_candidate["predicted_vor"] = expert_candidate["predicted_points"] - expert_candidate["replacement_points"]
        expert_candidate["predicted_ceiling_points"] = np.maximum(
            expert_candidate["predicted_ceiling_points"].to_numpy(),
            expert_candidate["predicted_points"].to_numpy(),
        )
        expert_pool_metrics = evaluate_draft_pool(
            expert_candidate,
            predicted_col="predicted_points",
            actual_col="fantasy_points_total",
            scope=position,
        )
        if expert_pool_metrics["top_pool_mae"] <= evaluate_draft_pool(
            combined_validation,
            predicted_col="predicted_points",
            actual_col="fantasy_points_total",
            scope=position,
        )["top_pool_mae"] * 1.03:
            combined_validation = expert_candidate
            train_combined_frame["archetype_expert_adjustment"] = 0.35 * apply_archetype_expert_adjustments(
                train_combined_frame,
                archetype_expert_bundle,
            )
            train_combined_frame["predicted_points"] = (
                train_combined_frame["predicted_points"] + train_combined_frame["archetype_expert_adjustment"]
            ).clip(lower=0.0)
            train_combined_frame["predicted_vor"] = train_combined_frame["predicted_points"] - train_combined_frame["replacement_points"]
            train_combined_frame["predicted_ceiling_points"] = np.maximum(
                train_combined_frame["predicted_ceiling_points"].to_numpy(),
                train_combined_frame["predicted_points"].to_numpy(),
            )
        else:
            archetype_expert_bundle = {}
            combined_validation["archetype_expert_adjustment"] = 0.0
            train_combined_frame["archetype_expert_adjustment"] = 0.0

        point_calibration_bundle = fit_projection_calibration_bundle(
            combined_validation,
            predicted_col="predicted_points",
            actual_col="fantasy_points_total",
            scope=position,
            mode="points",
        )
        combined_validation["predicted_points"] = apply_regression_calibration_bundle(
            combined_validation["predicted_points"].to_numpy(),
            point_calibration_bundle,
        )
        train_combined_frame["predicted_points"] = apply_regression_calibration_bundle(
            train_combined_frame["predicted_points"].to_numpy(),
            point_calibration_bundle,
        )
        combined_validation["predicted_vor"] = combined_validation["predicted_points"] - combined_validation["replacement_points"]
        train_combined_frame["predicted_vor"] = train_combined_frame["predicted_points"] - train_combined_frame["replacement_points"]
        combined_validation["predicted_ceiling_points_direct"] = np.maximum(
            combined_validation["predicted_points"].to_numpy(),
            combined_validation["predicted_ceiling_points_direct"].to_numpy(),
        )
        train_combined_frame["predicted_ceiling_points_direct"] = np.maximum(
            train_combined_frame["predicted_points"].to_numpy(),
            train_combined_frame["predicted_ceiling_points_direct"].to_numpy(),
        )
        combined_validation["ceiling_points_target"] = build_ceiling_points_target(position_valid_df).to_numpy()
        ceiling_calibration_bundle = fit_projection_calibration_bundle(
            combined_validation,
            predicted_col="predicted_ceiling_points_direct",
            actual_col="ceiling_points_target",
            scope=position,
            mode="ceiling",
        )
        combined_validation["predicted_ceiling_points_direct"] = np.maximum(
            combined_validation["predicted_points"].to_numpy(),
            apply_regression_calibration_bundle(
                combined_validation["predicted_ceiling_points_direct"].to_numpy(),
                ceiling_calibration_bundle,
            ),
        )
        train_combined_frame["predicted_ceiling_points_direct"] = np.maximum(
            train_combined_frame["predicted_points"].to_numpy(),
            apply_regression_calibration_bundle(
                train_combined_frame["predicted_ceiling_points_direct"].to_numpy(),
                ceiling_calibration_bundle,
            ),
        )
        combined_validation["predicted_ceiling_points"] = np.maximum(
            combined_validation["predicted_points"].to_numpy(),
            combined_validation["predicted_ceiling_points_direct"].to_numpy(),
        )
        train_combined_frame["predicted_ceiling_points"] = np.maximum(
            train_combined_frame["predicted_points"].to_numpy(),
            train_combined_frame["predicted_ceiling_points_direct"].to_numpy(),
        )

        distribution_bundle = fit_point_distribution_table(combined_validation)
        point_quantiles_valid = apply_point_distribution(
            combined_validation,
            combined_validation["predicted_points"],
            distribution_bundle,
        )
        for column, values in point_quantiles_valid.items():
            combined_validation[column] = values
        combined_validation["predicted_ceiling_points"] = np.maximum(
            combined_validation["predicted_points"].to_numpy(),
            (
                (0.60 * combined_validation["predicted_ceiling_points_direct"].to_numpy())
                + (0.40 * combined_validation["predicted_points_p90"].fillna(combined_validation["predicted_points"]).to_numpy())
            ),
        )
        games_quantiles_valid = bucket_probabilities_to_quantiles(valid_bucket_probabilities)
        for quantile, values in games_quantiles_valid.items():
            combined_validation[f"predicted_games_p{int(quantile * 100)}"] = values

        combined_validation["ranker_model"] = f"Stack[{ranker_summary}]"
        combined_validation["ranker_bonus_scale"] = ranker_bonus_scale
        validation_outputs.append(combined_validation)

        selected_point_metrics = evaluate_predictions(
            combined_validation["fantasy_points_total"],
            combined_validation["predicted_points"].to_numpy(),
        )
        selected_vor_metrics = evaluate_predictions(
            combined_validation["vor"],
            combined_validation["predicted_vor"].to_numpy(),
        )
        comparison_rows.append(
            {
                "scope": position,
                "model": f"SelectedProduction[Stack[{combo_summary}]+Stack[{ranker_summary}]]",
                "point_rmse": selected_point_metrics["rmse"],
                "point_mae": selected_point_metrics["mae"],
                "point_corr": selected_point_metrics["corr"],
                "vor_rmse": selected_vor_metrics["rmse"],
                "vor_mae": selected_vor_metrics["mae"],
                "vor_corr": selected_vor_metrics["corr"],
                **evaluate_draft_pool(combined_validation, "predicted_points", "fantasy_points_total", position),
                **evaluate_tail_metrics(combined_validation, position),
            }
        )

        final_availability_candidates = fit_availability_candidate_model_set(
            position_final_train_df,
            model_features,
            numeric_features,
            categorical_features,
            final_train_weights,
        )
        final_ppg_candidates = fit_candidate_model_set(
            position_final_train_df,
            model_features,
            "fantasy_points_per_game",
            numeric_features,
            categorical_features,
            final_train_weights,
        )
        final_combined_train, final_bucket_probabilities = combine_combo_predictions(
            position_final_train_df,
            final_availability_candidates,
            final_ppg_candidates,
            final_priors,
        )
        final_combined_train["availability_downside_probability"] = apply_grouped_probability_calibrator(
            final_combined_train["availability_downside_probability"].to_numpy(),
            position_final_train_df,
            availability_calibration_bundle,
        )

        final_elite_bundle = {
            "candidate_models": fit_probability_candidate_models(
                position_final_train_df,
                model_features,
                numeric_features,
                categorical_features,
                build_elite_finish_target(position_final_train_df),
                final_train_weights,
            ),
            "stack_weights": elite_bundle["stack_weights"],
            "calibration_bundle": elite_bundle["calibration_bundle"],
            "target_builder": build_elite_finish_target,
        }
        final_top_tier_bundle = {
            "candidate_models": fit_probability_candidate_models(
                position_final_train_df,
                model_features,
                numeric_features,
                categorical_features,
                build_top_tier_finish_target(position_final_train_df),
                final_train_weights,
            ),
            "stack_weights": top_tier_bundle["stack_weights"],
            "calibration_bundle": top_tier_bundle["calibration_bundle"],
            "target_builder": build_top_tier_finish_target,
        }
        final_major_jump_bundle = {
            "candidate_models": fit_probability_candidate_models(
                position_final_train_df,
                model_features,
                numeric_features,
                categorical_features,
                build_major_jump_target(position_final_train_df),
                final_train_weights,
            ),
            "stack_weights": major_jump_bundle["stack_weights"],
            "calibration_bundle": major_jump_bundle["calibration_bundle"],
            "target_builder": build_major_jump_target,
        }
        final_role_expansion_bundle = {
            "candidate_models": fit_probability_candidate_models(
                position_final_train_df,
                model_features,
                numeric_features,
                categorical_features,
                build_role_expansion_target(position_final_train_df),
                final_train_weights,
            ),
            "stack_weights": role_expansion_bundle["stack_weights"],
            "calibration_bundle": role_expansion_bundle["calibration_bundle"],
            "target_builder": build_role_expansion_target,
        }
        final_collapse_bundle = {
            "candidate_models": fit_probability_candidate_models(
                position_final_train_df,
                model_features,
                numeric_features,
                categorical_features,
                build_collapse_target(position_final_train_df),
                final_train_weights,
            ),
            "stack_weights": collapse_bundle["stack_weights"],
            "calibration_bundle": collapse_bundle["calibration_bundle"],
            "target_builder": build_collapse_target,
        }
        final_combined_train["elite_finish_probability"] = predict_event_bundle(
            final_elite_bundle,
            position_final_train_df[model_features],
        )
        final_combined_train["top_tier_finish_probability"] = predict_event_bundle(
            final_top_tier_bundle,
            position_final_train_df[model_features],
        )
        final_combined_train["major_jump_probability"] = predict_event_bundle(
            final_major_jump_bundle,
            position_final_train_df[model_features],
        )
        final_combined_train["role_expansion_probability"] = predict_event_bundle(
            final_role_expansion_bundle,
            position_final_train_df[model_features],
        )
        final_combined_train["breakout_probability"] = np.clip(
            (0.30 * final_combined_train["top_tier_finish_probability"])
            + (0.45 * final_combined_train["major_jump_probability"])
            + (0.25 * final_combined_train["role_expansion_probability"]),
            0.0,
            1.0,
        )
        final_combined_train["collapse_probability"] = predict_event_bundle(
            final_collapse_bundle,
            position_final_train_df[model_features],
        )

        final_ceiling_features = build_ceiling_feature_frame(final_combined_train, model_features)
        final_ceiling_frame = final_ceiling_features.copy()
        final_ceiling_frame["ceiling_points_target"] = build_ceiling_points_target(position_final_train_df)
        final_ceiling_weights = final_train_weights * (
            1.0
            + (5.0 * build_top_tier_finish_target(position_final_train_df).to_numpy(dtype=float))
            + (4.0 * build_major_jump_target(position_final_train_df).to_numpy(dtype=float))
            + (2.5 * build_role_expansion_target(position_final_train_df).to_numpy(dtype=float))
        )
        final_ceiling_candidates = fit_candidate_model_set(
            final_ceiling_frame,
            ceiling_feature_columns,
            "ceiling_points_target",
            ceiling_numeric_features,
            categorical_features,
            final_ceiling_weights,
        )
        final_combined_train["predicted_ceiling_points_direct"] = weighted_average_series(
            {
                name: np.clip(
                    model.predict(final_ceiling_features[ceiling_feature_columns]),
                    final_combined_train["predicted_points"].to_numpy(),
                    None,
                )
                for name, model in final_ceiling_candidates.items()
            },
            ceiling_weights,
        )
        final_combined_train["predicted_ceiling_points"] = np.maximum(
            final_combined_train["predicted_ceiling_points_direct"].to_numpy(),
            final_combined_train["predicted_points"].to_numpy(),
        )
        final_upside_adjustments = apply_upside_archetype_routing(final_combined_train, upside_archetype_bundle)
        final_combined_train["upside_archetype_ceiling_adjustment"] = final_upside_adjustments["ceiling_points_adjustment"]
        final_combined_train["predicted_ceiling_points_direct"] = np.maximum(
            final_combined_train["predicted_points"].to_numpy(),
            final_combined_train["predicted_ceiling_points_direct"].to_numpy()
            + final_combined_train["upside_archetype_ceiling_adjustment"].to_numpy(),
        )
        final_combined_train["predicted_ceiling_points"] = np.maximum(
            final_combined_train["predicted_points"].to_numpy(),
            final_combined_train["predicted_ceiling_points"].to_numpy()
            + final_combined_train["upside_archetype_ceiling_adjustment"].to_numpy(),
        )
        final_combined_train["top_tier_finish_probability"] = np.clip(
            final_combined_train["top_tier_finish_probability"].to_numpy()
            + final_upside_adjustments["top_tier_probability_adjustment"],
            0.0,
            1.0,
        )
        final_combined_train["major_jump_probability"] = np.clip(
            final_combined_train["major_jump_probability"].to_numpy()
            + final_upside_adjustments["major_jump_probability_adjustment"],
            0.0,
            1.0,
        )
        final_combined_train["role_expansion_probability"] = np.clip(
            final_combined_train["role_expansion_probability"].to_numpy()
            + final_upside_adjustments["role_expansion_probability_adjustment"],
            0.0,
            1.0,
        )
        final_combined_train["breakout_probability"] = np.clip(
            (0.30 * final_combined_train["top_tier_finish_probability"])
            + (0.45 * final_combined_train["major_jump_probability"])
            + (0.25 * final_combined_train["role_expansion_probability"]),
            0.0,
            1.0,
        )
        if archetype_expert_bundle:
            final_combined_train["archetype_expert_adjustment"] = 0.35 * apply_archetype_expert_adjustments(
                position_final_train_df,
                archetype_expert_bundle,
            )
            final_combined_train["predicted_points"] = (
                final_combined_train["predicted_points"] + final_combined_train["archetype_expert_adjustment"]
            ).clip(lower=0.0)
            final_combined_train["predicted_vor"] = final_combined_train["predicted_points"] - final_combined_train["replacement_points"]
            final_combined_train["predicted_ceiling_points"] = np.maximum(
                final_combined_train["predicted_ceiling_points"].to_numpy(),
                final_combined_train["predicted_points"].to_numpy(),
            )
        else:
            final_combined_train["archetype_expert_adjustment"] = 0.0

        final_combined_train["predicted_points"] = apply_regression_calibration_bundle(
            final_combined_train["predicted_points"].to_numpy(),
            point_calibration_bundle,
        )
        final_combined_train["predicted_vor"] = final_combined_train["predicted_points"] - final_combined_train["replacement_points"]
        final_combined_train["predicted_ceiling_points_direct"] = np.maximum(
            final_combined_train["predicted_points"].to_numpy(),
            apply_regression_calibration_bundle(
                final_combined_train["predicted_ceiling_points_direct"].to_numpy(),
                ceiling_calibration_bundle,
            ),
        )
        final_combined_train["predicted_ceiling_points"] = np.maximum(
            final_combined_train["predicted_points"].to_numpy(),
            final_combined_train["predicted_ceiling_points_direct"].to_numpy(),
        )

        final_ranker_models: Dict[str, Pipeline] = {}
        if ranker_weights_map:
            final_ranker_target = build_ranker_target(position_final_train_df)
            final_ranker_weights = final_train_weights * (1.0 + (12.0 * final_ranker_target.to_numpy()))
            final_ranker_features = build_ranker_feature_frame(final_combined_train, model_features)
            for ranker_name in ranker_weights_map:
                final_ranker_model = build_model_by_name(ranker_name, ranker_numeric_features, categorical_features)
                final_ranker_model.fit(
                    final_ranker_features[ranker_feature_columns],
                    final_ranker_target,
                    model__sample_weight=final_ranker_weights,
                )
                final_ranker_models[ranker_name] = final_ranker_model

        position_models[position] = {
            "combo_weights": combo_weights,
            "combo_models": [
                {
                    "name": combo_name,
                    "availability_model_name": combo_meta["availability_model_name"],
                    "ppg_model_name": combo_meta["ppg_model_name"],
                    "availability_model": final_availability_candidates[combo_meta["availability_model_name"]],
                    "ppg_model": final_ppg_candidates[combo_meta["ppg_model_name"]],
                    "weight": combo_weights.get(combo_name, 0.0),
                    **combo_calibration_info[combo_name],
                }
                for combo_name, combo_meta in combo_outputs.items()
            ],
            "availability_calibration_bundle": availability_calibration_bundle,
            "elite_bundle": final_elite_bundle,
            "top_tier_bundle": final_top_tier_bundle,
            "major_jump_bundle": final_major_jump_bundle,
            "role_expansion_bundle": final_role_expansion_bundle,
            "collapse_bundle": final_collapse_bundle,
            "ceiling_models": final_ceiling_candidates,
            "ceiling_weights": ceiling_weights,
            "ceiling_feature_columns": ceiling_feature_columns,
            "ceiling_numeric_features": ceiling_numeric_features,
            "ranker_models": final_ranker_models,
            "ranker_weights": ranker_weights_map,
            "ranker_model_name": f"Stack[{ranker_summary}]",
            "ranker_bonus_scale": ranker_bonus_scale,
            "selected_model": f"Stack[{combo_summary}]",
            "ranker_feature_columns": ranker_feature_columns,
            "ranker_numeric_features": ranker_numeric_features,
            "projection_priors": final_priors,
            "point_distribution_bundle": distribution_bundle,
            "archetype_expert_bundle": archetype_expert_bundle,
            "upside_archetype_bundle": upside_archetype_bundle,
            "point_calibration_bundle": point_calibration_bundle,
            "ceiling_calibration_bundle": ceiling_calibration_bundle,
        }

    valid_output = pd.concat(validation_outputs, ignore_index=True).sort_values(
        ["season", "position", "player_name"]
    ).reset_index(drop=True)
    overall_prod_point_metrics = evaluate_predictions(
        valid_output["fantasy_points_total"],
        valid_output["predicted_points"].to_numpy(),
    )
    overall_prod_vor_metrics = evaluate_predictions(
        valid_output["vor"],
        valid_output["predicted_vor"].to_numpy(),
    )
    comparison_rows.append(
        {
            "scope": "ALL",
            "model": "SelectedProduction",
            "point_rmse": overall_prod_point_metrics["rmse"],
            "point_mae": overall_prod_point_metrics["mae"],
            "point_corr": overall_prod_point_metrics["corr"],
            "vor_rmse": overall_prod_vor_metrics["rmse"],
            "vor_mae": overall_prod_vor_metrics["mae"],
            "vor_corr": overall_prod_vor_metrics["corr"],
            **evaluate_draft_pool(valid_output, "predicted_points", "fantasy_points_total", "ALL"),
            **evaluate_tail_metrics(valid_output, "ALL"),
        }
    )
    comparison = pd.DataFrame(comparison_rows).sort_values(["scope", "model"]).reset_index(drop=True)
    model_bundle = {
        "position_models": position_models,
        "features": model_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "projection_priors": final_priors,
        "selected_model_summary": ", ".join(
            f"{position}={bundle['selected_model']}+{bundle['ranker_model_name']}" for position, bundle in position_models.items()
        ),
    }
    return model_bundle, comparison, valid_output


def estimate_replacement_levels(feature_df: pd.DataFrame, target_year: int, lookback: int = 3) -> Dict[str, float]:
    recent = feature_df[
        (feature_df["season"] >= target_year - lookback) & (feature_df["season"] < target_year)
    ][["season", "position", "replacement_points"]].drop_duplicates()
    replacement_projection = recent.groupby("position")["replacement_points"].mean().to_dict()
    for position in REPLACEMENT_RANKS:
        replacement_projection.setdefault(position, recent["replacement_points"].median())
    return replacement_projection


def calculate_league_starter_counts(
    projections_df: pd.DataFrame,
    league_size: int = LEAGUE_SIZE,
    value_column: str = "predicted_points",
) -> Dict[str, int]:
    starter_counts = {position: PRIMARY_STARTERS[position] * league_size for position in PRIMARY_STARTERS}

    flex_pool_frames: List[pd.DataFrame] = []
    for position in FLEX_POSITIONS:
        position_pool = projections_df[projections_df["position"] == position].sort_values(
            value_column, ascending=False
        )
        flex_candidates = position_pool.iloc[starter_counts[position] :][["player_key", "position", value_column]]
        flex_pool_frames.append(flex_candidates)

    flex_pool = pd.concat(flex_pool_frames, ignore_index=True).sort_values(value_column, ascending=False)
    flex_winners = flex_pool.head(league_size * FLEX_SLOTS)
    flex_counts = flex_winners["position"].value_counts().to_dict()

    effective_starter_counts = starter_counts.copy()
    for position, extra_starters in flex_counts.items():
        effective_starter_counts[position] += int(extra_starters)

    return effective_starter_counts


def calculate_position_baselines(
    projections_df: pd.DataFrame, cutoff_counts: Dict[str, int], value_column: str
) -> Dict[str, float]:
    baselines: Dict[str, float] = {}
    for position, cutoff in cutoff_counts.items():
        position_pool = projections_df[projections_df["position"] == position].sort_values(
            value_column, ascending=False
        ).reset_index(drop=True)
        if position_pool.empty:
            baselines[position] = 0.0
            continue
        cutoff_index = min(max(cutoff, 1) - 1, len(position_pool) - 1)
        baselines[position] = float(position_pool.loc[cutoff_index, value_column])
    return baselines


def project_2024_players(feature_df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    target_df = feature_df[feature_df["season"] == 2024].copy()
    feature_cols = model_bundle["features"]
    replacement_projection = estimate_replacement_levels(feature_df, target_year=2024, lookback=3)
    target_df["replacement_projection"] = target_df["position"].map(replacement_projection)

    target_df["predicted_games_raw"] = np.nan
    target_df["predicted_ppg_raw"] = np.nan
    target_df["predicted_vor_uncalibrated"] = 0.0
    target_df["predicted_vor"] = 0.0
    target_df["availability_downside_probability"] = 0.0
    target_df["elite_finish_probability"] = 0.0
    target_df["top_tier_finish_probability"] = 0.0
    target_df["major_jump_probability"] = 0.0
    target_df["role_expansion_probability"] = 0.0
    target_df["breakout_probability"] = 0.0
    target_df["collapse_probability"] = 0.0
    target_df["predicted_ceiling_points_direct"] = 0.0
    target_df["predicted_ceiling_points"] = 0.0
    target_df["draft_ranker_raw"] = 0.0
    target_df["draft_ranker_bonus"] = 0.0
    target_df["selected_model"] = ""
    target_df["archetype_expert_adjustment"] = 0.0
    target_df["upside_archetype_ceiling_adjustment"] = 0.0
    for position, position_bundle in model_bundle["position_models"].items():
        position_mask = target_df["position"] == position
        position_rows = target_df.loc[position_mask, feature_cols]
        combo_games_raw: Dict[str, np.ndarray] = {}
        combo_ppg_raw: Dict[str, np.ndarray] = {}
        combo_games: Dict[str, np.ndarray] = {}
        combo_ppg: Dict[str, np.ndarray] = {}
        combo_points_uncalibrated: Dict[str, np.ndarray] = {}
        combo_points: Dict[str, np.ndarray] = {}
        combo_downside: Dict[str, np.ndarray] = {}
        combo_bucket_probabilities: Dict[str, np.ndarray] = {}
        combo_adjustment_columns: Dict[str, Dict[str, np.ndarray]] = {}
        combo_weights = position_bundle["combo_weights"]

        for combo_model in position_bundle["combo_models"]:
            combo_name = combo_model["name"]
            games_raw, downside_raw, bucket_probabilities = predict_availability_distribution(
                combo_model["availability_model"],
                position_rows,
            )
            ppg_raw = combo_model["ppg_model"].predict(position_rows)
            games_adj, ppg_adj, adjustment_columns = apply_projection_priors(
                target_df.loc[position_mask],
                predicted_games=games_raw,
                predicted_ppg=ppg_raw,
                projection_priors=position_bundle.get("projection_priors", model_bundle["projection_priors"]),
            )
            points_uncalibrated = games_adj * ppg_adj
            points = np.clip(
                combo_model["calibration_intercept"] + (combo_model["calibration_slope"] * points_uncalibrated),
                0.0,
                None,
            )

            combo_games_raw[combo_name] = games_raw
            combo_ppg_raw[combo_name] = ppg_raw
            combo_games[combo_name] = games_adj
            combo_ppg[combo_name] = ppg_adj
            combo_points_uncalibrated[combo_name] = points_uncalibrated
            combo_points[combo_name] = points
            combo_downside[combo_name] = downside_raw
            combo_bucket_probabilities[combo_name] = bucket_probabilities
            combo_adjustment_columns[combo_name] = adjustment_columns

        combined_bucket_probabilities = weighted_average_series(combo_bucket_probabilities, combo_weights)
        target_df.loc[position_mask, "predicted_games_raw"] = weighted_average_series(combo_games_raw, combo_weights)
        target_df.loc[position_mask, "predicted_ppg_raw"] = weighted_average_series(combo_ppg_raw, combo_weights)
        target_df.loc[position_mask, "predicted_games"] = weighted_average_series(combo_games, combo_weights)
        target_df.loc[position_mask, "predicted_ppg"] = weighted_average_series(combo_ppg, combo_weights)
        target_df.loc[position_mask, "predicted_points_uncalibrated"] = weighted_average_series(
            combo_points_uncalibrated,
            combo_weights,
        )
        target_df.loc[position_mask, "predicted_points"] = weighted_average_series(combo_points, combo_weights)
        target_df.loc[position_mask, "predicted_vor_uncalibrated"] = (
            target_df.loc[position_mask, "predicted_points_uncalibrated"].to_numpy()
            - target_df.loc[position_mask, "replacement_projection"].to_numpy()
        )
        target_df.loc[position_mask, "predicted_vor"] = (
            target_df.loc[position_mask, "predicted_points"].to_numpy()
            - target_df.loc[position_mask, "replacement_projection"].to_numpy()
        )
        target_df.loc[position_mask, "availability_downside_probability"] = apply_grouped_probability_calibrator(
            weighted_average_series(combo_downside, combo_weights),
            target_df.loc[position_mask],
            position_bundle.get("availability_calibration_bundle", {}),
        )
        for adjustment_column in [
            "rookie_games_adjustment",
            "rookie_ppg_adjustment",
            "comeback_games_adjustment",
            "comeback_ppg_adjustment",
            "thin_history_games_adjustment",
            "thin_history_ppg_adjustment",
        ]:
            target_df.loc[position_mask, adjustment_column] = weighted_average_series(
                {name: cols[adjustment_column] for name, cols in combo_adjustment_columns.items()},
                combo_weights,
            )

        games_quantiles = bucket_probabilities_to_quantiles(combined_bucket_probabilities)
        for quantile, values in games_quantiles.items():
            target_df.loc[position_mask, f"predicted_games_p{int(quantile * 100)}"] = values

        target_df.loc[position_mask, "elite_finish_probability"] = predict_event_bundle(
            position_bundle["elite_bundle"],
            position_rows,
        )
        target_df.loc[position_mask, "top_tier_finish_probability"] = predict_event_bundle(
            position_bundle["top_tier_bundle"],
            position_rows,
        )
        target_df.loc[position_mask, "major_jump_probability"] = predict_event_bundle(
            position_bundle["major_jump_bundle"],
            position_rows,
        )
        target_df.loc[position_mask, "role_expansion_probability"] = predict_event_bundle(
            position_bundle["role_expansion_bundle"],
            position_rows,
        )
        target_df.loc[position_mask, "collapse_probability"] = predict_event_bundle(
            position_bundle["collapse_bundle"],
            position_rows,
        )
        target_df.loc[position_mask, "breakout_probability"] = np.clip(
            (0.30 * target_df.loc[position_mask, "top_tier_finish_probability"])
            + (0.45 * target_df.loc[position_mask, "major_jump_probability"])
            + (0.25 * target_df.loc[position_mask, "role_expansion_probability"]),
            0.0,
            1.0,
        )
        ceiling_features = build_ceiling_feature_frame(target_df.loc[position_mask].copy(), feature_cols)
        direct_ceiling_predictions = {
            model_name: np.clip(
                model.predict(ceiling_features[position_bundle["ceiling_feature_columns"]]),
                target_df.loc[position_mask, "predicted_points"].to_numpy(),
                None,
            )
            for model_name, model in position_bundle.get("ceiling_models", {}).items()
        }
        if direct_ceiling_predictions:
            target_df.loc[position_mask, "predicted_ceiling_points_direct"] = weighted_average_series(
                direct_ceiling_predictions,
                position_bundle.get("ceiling_weights", {}),
            )
        else:
            target_df.loc[position_mask, "predicted_ceiling_points_direct"] = target_df.loc[position_mask, "predicted_points"]

        upside_adjustments = apply_upside_archetype_routing(
            target_df.loc[position_mask].copy(),
            position_bundle.get("upside_archetype_bundle", {}),
        )
        target_df.loc[position_mask, "upside_archetype_ceiling_adjustment"] = upside_adjustments["ceiling_points_adjustment"]
        target_df.loc[position_mask, "predicted_ceiling_points_direct"] = np.maximum(
            target_df.loc[position_mask, "predicted_points"].to_numpy(),
            target_df.loc[position_mask, "predicted_ceiling_points_direct"].to_numpy()
            + target_df.loc[position_mask, "upside_archetype_ceiling_adjustment"].to_numpy(),
        )
        target_df.loc[position_mask, "top_tier_finish_probability"] = np.clip(
            target_df.loc[position_mask, "top_tier_finish_probability"].to_numpy()
            + upside_adjustments["top_tier_probability_adjustment"],
            0.0,
            1.0,
        )
        target_df.loc[position_mask, "major_jump_probability"] = np.clip(
            target_df.loc[position_mask, "major_jump_probability"].to_numpy()
            + upside_adjustments["major_jump_probability_adjustment"],
            0.0,
            1.0,
        )
        target_df.loc[position_mask, "role_expansion_probability"] = np.clip(
            target_df.loc[position_mask, "role_expansion_probability"].to_numpy()
            + upside_adjustments["role_expansion_probability_adjustment"],
            0.0,
            1.0,
        )
        target_df.loc[position_mask, "breakout_probability"] = np.clip(
            (0.30 * target_df.loc[position_mask, "top_tier_finish_probability"])
            + (0.45 * target_df.loc[position_mask, "major_jump_probability"])
            + (0.25 * target_df.loc[position_mask, "role_expansion_probability"]),
            0.0,
            1.0,
        )
        target_df.loc[position_mask, "predicted_ceiling_points"] = np.maximum(
            target_df.loc[position_mask, "predicted_points"].to_numpy(),
            target_df.loc[position_mask, "predicted_ceiling_points_direct"].to_numpy(),
        )
        target_df.loc[position_mask, "selected_model"] = position_bundle["selected_model"]

    target_df["predicted_vor_uncalibrated"] = target_df["predicted_points_uncalibrated"] - target_df["replacement_projection"]
    target_df["predicted_vor"] = target_df["predicted_points"] - target_df["replacement_projection"]

    for position, position_bundle in model_bundle["position_models"].items():
        position_mask = target_df["position"] == position
        position_target_df = target_df.loc[position_mask].copy()
        expert_adjustment = 0.0
        if position_bundle.get("archetype_expert_bundle"):
            expert_adjustment = 0.35 * apply_archetype_expert_adjustments(
                position_target_df,
                position_bundle["archetype_expert_bundle"],
            )
        target_df.loc[position_mask, "archetype_expert_adjustment"] = expert_adjustment
        target_df.loc[position_mask, "predicted_points"] = (
            target_df.loc[position_mask, "predicted_points"] + expert_adjustment
        ).clip(lower=0.0)
        target_df.loc[position_mask, "predicted_points"] = apply_regression_calibration_bundle(
            target_df.loc[position_mask, "predicted_points"].to_numpy(),
            position_bundle.get("point_calibration_bundle", {}),
        )
        target_df.loc[position_mask, "predicted_vor"] = (
            target_df.loc[position_mask, "predicted_points"] - target_df.loc[position_mask, "replacement_projection"]
        )
        target_df.loc[position_mask, "predicted_ceiling_points_direct"] = np.maximum(
            target_df.loc[position_mask, "predicted_points"].to_numpy(),
            apply_regression_calibration_bundle(
                target_df.loc[position_mask, "predicted_ceiling_points_direct"].to_numpy(),
                position_bundle.get("ceiling_calibration_bundle", {}),
            ),
        )
        point_quantiles = apply_point_distribution(
            target_df.loc[position_mask],
            target_df.loc[position_mask, "predicted_points"],
            position_bundle.get("point_distribution_bundle", {}),
        )
        for column, values in point_quantiles.items():
            target_df.loc[position_mask, column] = values
        target_df.loc[position_mask, "predicted_ceiling_points"] = np.maximum(
            target_df.loc[position_mask, "predicted_points"].to_numpy(),
            (
                (0.60 * target_df.loc[position_mask, "predicted_ceiling_points_direct"].to_numpy())
                + (0.40 * target_df.loc[position_mask, "predicted_points_p90"].fillna(target_df.loc[position_mask, "predicted_points"]).to_numpy())
            ),
        )

        if position_bundle["ranker_models"]:
            ranker_features = build_ranker_feature_frame(target_df.loc[position_mask].copy(), feature_cols)
            rank_scores = {
                ranker_name: model.predict(ranker_features[position_bundle["ranker_feature_columns"]])
                for ranker_name, model in position_bundle["ranker_models"].items()
            }
            raw_rank_scores = weighted_average_series(rank_scores, position_bundle["ranker_weights"])
            target_df.loc[position_mask, "draft_ranker_raw"] = raw_rank_scores
            target_df.loc[position_mask, "draft_ranker_bonus"] = (
                position_bundle["ranker_bonus_scale"] * standardize_array(raw_rank_scores)
            )

    target_df["draft_rank_score"] = (target_df["predicted_points"] + target_df["draft_ranker_bonus"]).clip(lower=0.0)

    league_starter_counts = calculate_league_starter_counts(
        target_df,
        league_size=LEAGUE_SIZE,
        value_column="draft_rank_score",
    )
    starter_baselines = calculate_position_baselines(
        target_df,
        cutoff_counts=league_starter_counts,
        value_column="draft_rank_score",
    )
    target_df["league_starter_baseline"] = target_df["position"].map(starter_baselines)
    target_df["starter_surplus_points"] = target_df["draft_rank_score"] - target_df["league_starter_baseline"]
    target_df["replacement_surplus_points"] = target_df["predicted_points"] - target_df["replacement_projection"]

    top_points_by_position = target_df.groupby("position")["draft_rank_score"].max().to_dict()
    target_df["scarcity"] = target_df.apply(
        lambda row: (
            (top_points_by_position[row["position"]] - row["league_starter_baseline"])
            / max(row["league_starter_baseline"], 1.0)
        ),
        axis=1,
    )
    target_df["scarcity_weight"] = 0.0
    target_df["upside_score"] = (
        (1.10 * target_df["elite_finish_probability"])
        + (0.90 * target_df["top_tier_finish_probability"])
        + (0.80 * target_df["major_jump_probability"])
        + (0.55 * target_df["role_expansion_probability"])
    )
    target_df["downside_score"] = (
        target_df["collapse_probability"] + target_df["availability_downside_probability"]
    )
    target_df["position_ceiling_blend"] = target_df["position"].map(POSITION_CEILING_BLEND).fillna(0.45)
    target_df["position_ceiling_draft_weight"] = target_df["position"].map(POSITION_CEILING_DRAFT_WEIGHT).fillna(0.12)
    target_df["position_public_ceiling_weight"] = target_df["position"].map(POSITION_PUBLIC_CEILING_WEIGHT).fillna(0.10)
    target_df["base_ceiling_quantile_points"] = (
        (0.30 * target_df["predicted_points_p75"].fillna(target_df["predicted_points"]))
        + (0.70 * target_df["predicted_points_p90"].fillna(target_df["predicted_points"]))
    )
    target_df["ceiling_case_points"] = (
        (target_df["position_ceiling_blend"] * target_df["predicted_ceiling_points"].fillna(target_df["predicted_points"]))
        + ((1.0 - target_df["position_ceiling_blend"]) * target_df["base_ceiling_quantile_points"])
    )
    target_df["floor_case_points"] = (
        (0.60 * target_df["predicted_points_p10"].fillna(target_df["predicted_points"]))
        + (0.40 * target_df["predicted_points_p25"].fillna(target_df["predicted_points"]))
    )
    target_df["position_upside_multiplier"] = target_df["position"].map(POSITION_UPSIDE_MULTIPLIER).fillna(1.0)
    target_df["position_downside_multiplier"] = target_df["position"].map(POSITION_DOWNSIDE_MULTIPLIER).fillna(1.0)
    target_df["archetype_upside_bonus"] = np.select(
        [
            target_df["position_archetype"].eq("dual_threat_qb"),
            target_df["position_archetype"].eq("bell_cow_rb"),
            target_df["position_archetype"].eq("receiving_rb"),
            target_df["position_archetype"].eq("alpha_wr"),
            target_df["position_archetype"].eq("featured_te"),
        ],
        [0.18, 0.14, 0.10, 0.14, 0.08],
        default=0.0,
    )
    target_df["league_winner_score"] = (
        (0.80 * target_df["elite_finish_probability"])
        + (0.65 * target_df["top_tier_finish_probability"])
        + (0.65 * target_df["major_jump_probability"])
        + (0.35 * target_df["role_expansion_probability"])
        + (0.12 * target_df["is_second_year"])
        + (0.10 * target_df["injury_return_candidate"])
        + target_df["archetype_upside_bonus"]
    ).clip(lower=0.0)
    target_df["upside_case_gain"] = (
        target_df["ceiling_case_points"] - target_df["predicted_points"]
    ).clip(lower=0.0)
    target_df["downside_case_loss"] = (
        target_df["predicted_points"] - target_df["floor_case_points"]
    ).clip(lower=0.0)
    target_df["rb_opportunity_bonus_points"] = np.where(
        target_df["position"].eq("RB"),
        (
            10.0 * target_df["role_growth_signal"].clip(lower=0.0)
            + 8.0 * target_df["opportunity_shock_score"].clip(lower=0.0)
            + 20.0 * target_df["vacated_position_share"].clip(lower=0.0)
            + 8.0 * target_df["major_jump_probability"]
            + 6.0 * target_df["top_tier_finish_probability"]
            - 6.0 * target_df["same_role_competition_index"].clip(lower=0.0)
        ).clip(lower=0.0),
        0.0,
    )
    target_df["upside_tail_points"] = (
        target_df["upside_case_gain"]
        * target_df["position_upside_multiplier"]
        * (1.0 + (0.45 * target_df["league_winner_score"]))
    )
    target_df["downside_tail_points"] = (
        target_df["downside_case_loss"] * target_df["position_downside_multiplier"]
    )
    target_df["robust_expected_points"] = (
        target_df["predicted_points"]
        + (ROBUST_UPSIDE_WEIGHT * target_df["upside_tail_points"])
        - (ROBUST_DOWNSIDE_WEIGHT * target_df["downside_tail_points"])
    )
    target_df["ceiling_priority_points"] = (
        target_df["predicted_points"]
        + (0.60 * target_df["upside_tail_points"])
        - (0.12 * target_df["downside_tail_points"])
    )
    target_df["cvar_proxy_points"] = (
        0.5 * target_df["predicted_points_p10"].fillna(target_df["predicted_points"])
        + 0.5 * target_df["predicted_points_p25"].fillna(target_df["predicted_points"])
    )
    target_df["upside_bonus_points"] = (
        UPSIDE_BONUS_WEIGHT * ((0.70 * target_df["upside_score"]) + (0.30 * target_df["league_winner_score"]))
    ) + (0.08 * target_df["upside_tail_points"]) + target_df["rb_opportunity_bonus_points"]
    target_df["downside_penalty_points"] = (
        DOWNSIDE_PENALTY_WEIGHT * target_df["downside_score"]
    ) + (0.06 * target_df["downside_tail_points"])
    target_df["draft_value"] = (
        target_df["starter_surplus_points"].clip(lower=0.0)
        + (VOR_DRAFT_WEIGHT * target_df["predicted_vor"].clip(lower=0.0))
        + (RESERVE_DRAFT_WEIGHT * target_df["replacement_surplus_points"].clip(lower=0.0))
        + (0.45 * (target_df["robust_expected_points"] - target_df["replacement_projection"]).clip(lower=0.0))
        + (target_df["position_ceiling_draft_weight"] * (target_df["ceiling_priority_points"] - target_df["predicted_points"]).clip(lower=0.0))
        + target_df["draft_ranker_bonus"]
        + target_df["upside_bonus_points"]
        - target_df["downside_penalty_points"]
    )
    target_df["public_board_score"] = (
        target_df["draft_rank_score"]
        + (0.24 * target_df["upside_tail_points"])
        - (0.08 * target_df["downside_tail_points"])
        + (target_df["position_public_ceiling_weight"] * (target_df["ceiling_priority_points"] - target_df["predicted_points"]).clip(lower=0.0))
        + (0.45 * target_df["upside_bonus_points"])
        - (0.20 * target_df["downside_penalty_points"])
    )
    target_df["actual_points"] = target_df["fantasy_points_total"]
    target_df["actual_vor"] = target_df["vor"]
    return target_df


def get_roster_counts(roster: List[dict]) -> Dict[str, int]:
    counts = {position: 0 for position in REPLACEMENT_RANKS}
    for player in roster:
        counts[player["position"]] += 1
    return counts


def get_flex_filled(counts: Dict[str, int]) -> int:
    excess = (
        max(counts["RB"] - PRIMARY_STARTERS["RB"], 0)
        + max(counts["WR"] - PRIMARY_STARTERS["WR"], 0)
        + max(counts["TE"] - PRIMARY_STARTERS["TE"], 0)
    )
    return min(excess, FLEX_SLOTS)


def get_position_need_multiplier(team: TeamState, position: str, round_number: int) -> float:
    counts = get_roster_counts(team.roster)
    flex_filled = get_flex_filled(counts)
    starter_need = max(PRIMARY_STARTERS[position] - counts[position], 0)
    flex_need = max(FLEX_SLOTS - flex_filled, 0) if position in FLEX_POSITIONS else 0

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
        if position == "RB":
            multiplier = 1.12
        elif position == "WR":
            multiplier = 1.00
        else:
            multiplier = 0.98
    else:
        if position == "RB":
            multiplier = 0.92
        elif position == "WR":
            multiplier = 0.82
        else:
            multiplier = 0.75

    if position == "WR" and counts["WR"] >= 3 and round_number <= 8:
        multiplier *= 0.86
    if position == "RB" and counts["RB"] <= 1 and round_number <= 6:
        multiplier *= 1.08

    if counts[position] >= BENCH_SOFT_CAPS[position]:
        multiplier *= 0.65

    return multiplier


def is_candidate_eligible(team: TeamState, candidate: pd.Series, round_number: int) -> bool:
    counts = get_roster_counts(team.roster)
    position = candidate["position"]

    if position == "QB" and counts["QB"] >= 2:
        return False

    if position == "TE" and counts["TE"] >= 2:
        return False

    return True


def get_remaining_starter_demand(teams: List[TeamState], position: str) -> int:
    demand = 0
    for team in teams:
        counts = get_roster_counts(team.roster)
        demand += max(PRIMARY_STARTERS[position] - counts[position], 0)
    if position in FLEX_POSITIONS:
        open_flex = sum(max(FLEX_SLOTS - get_flex_filled(get_roster_counts(team.roster)), 0) for team in teams)
        demand += int(np.ceil(open_flex / len(FLEX_POSITIONS)))
    return max(demand, 1)


def get_position_pool_reference_points(
    available_df: pd.DataFrame, teams: List[TeamState], value_column: str = "public_board_score"
) -> Dict[str, float]:
    reference_points: Dict[str, float] = {}
    for position in REPLACEMENT_RANKS:
        position_pool = available_df[available_df["position"] == position].sort_values(
            value_column, ascending=False
        )
        if position_pool.empty:
            reference_points[position] = 0.0
            continue
        demand = get_remaining_starter_demand(teams, position)
        comparison_idx = min(demand - 1, len(position_pool) - 1)
        reference_points[position] = float(position_pool.iloc[comparison_idx][value_column])
    return reference_points


def get_stack_multiplier(team: TeamState, candidate: pd.Series) -> float:
    if candidate["position"] == "QB":
        matching_receivers = sum(
            1
            for player in team.roster
            if player["team"] == candidate["team"] and player["position"] in {"WR", "TE"}
        )
        return 1.0 + min(matching_receivers * STACK_BONUS, 0.10)

    if candidate["position"] in {"WR", "TE"}:
        matching_qbs = sum(
            1
            for player in team.roster
            if player["team"] == candidate["team"] and player["position"] == "QB"
        )
        return 1.0 + min(matching_qbs * STACK_BONUS, 0.10)

    return 1.0


def get_competition_multiplier(team: TeamState, candidate: pd.Series) -> float:
    same_team_same_position = sum(
        1
        for player in team.roster
        if player["team"] == candidate["team"] and player["position"] == candidate["position"]
    )
    penalty = min(same_team_same_position * COMPETITION_PENALTY, 0.16)
    return 1.0 - penalty


def get_candidate_risk_bucket(candidate: pd.Series | dict) -> str:
    downside_score = float(candidate.get("downside_score", 0.0))
    upside_tail = float(candidate.get("upside_tail_points", 0.0))
    if downside_score >= 0.75:
        return "fragile"
    if upside_tail >= 32.0 and downside_score <= 0.45:
        return "ceiling"
    return "balanced"


def get_submodular_multiplier(team: TeamState, candidate: pd.Series) -> float:
    same_archetype = sum(
        1
        for player in team.roster
        if player.get("position") == candidate["position"]
        and player.get("position_archetype") == candidate.get("position_archetype")
    )
    same_nfl_team = sum(1 for player in team.roster if player.get("team") == candidate["team"])
    candidate_risk_bucket = get_candidate_risk_bucket(candidate)
    same_risk_bucket = sum(
        1 for player in team.roster if get_candidate_risk_bucket(player) == candidate_risk_bucket
    )
    penalty = (
        (same_archetype * SUBMODULAR_ARCHETYPE_PENALTY)
        + (same_nfl_team * TEAM_DEPENDENCY_PENALTY)
        + (same_risk_bucket * RISK_BUCKET_PENALTY)
    )
    return float(np.clip(1.0 - penalty, 0.45, 1.05))


def get_position_preference_multiplier(team: TeamState, position: str) -> float:
    preference_map = team.position_pref_alpha or OPPONENT_PROFILE_ALPHAS.get(team.manager_profile, {})
    return float(preference_map.get(position, 1.0))


def softmax_choice(scores: np.ndarray, rng: np.random.Generator) -> int:
    standardized = (scores - scores.max()) / max(scores.std(ddof=0), 1.0)
    logits = standardized / SOFTMAX_TEMPERATURE
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    return int(rng.choice(np.arange(len(scores)), p=probabilities))


def stable_seed(*parts: object) -> int:
    hasher = blake2b(digest_size=8)
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"|")
    return int.from_bytes(hasher.digest(), byteorder="big", signed=False)


def clone_teams(teams: List[TeamState]) -> List[TeamState]:
    return [
        TeamState(
            name=team.name,
            roster=list(team.roster),
            manager_profile=team.manager_profile,
            position_pref_alpha=dict(team.position_pref_alpha),
        )
        for team in teams
    ]


def initialize_teams(league_size: int, rng: np.random.Generator) -> List[TeamState]:
    profile_names = list(OPPONENT_PROFILE_ALPHAS.keys())
    teams: List[TeamState] = []
    for index in range(1, league_size + 1):
        profile = str(rng.choice(profile_names))
        teams.append(
            TeamState(
                name=f"Team {index}",
                manager_profile=profile,
                position_pref_alpha=dict(OPPONENT_PROFILE_ALPHAS[profile]),
            )
        )
    return teams


def build_pick_slots(league_size: int, rounds: int) -> List[dict]:
    pick_slots: List[dict] = []
    overall_pick = 1
    for round_number in range(1, rounds + 1):
        order = list(range(league_size)) if round_number % 2 == 1 else list(range(league_size - 1, -1, -1))
        for pick_in_round, team_index in enumerate(order, start=1):
            pick_slots.append(
                {
                    "overall_pick": overall_pick,
                    "round": round_number,
                    "pick_in_round": pick_in_round,
                    "team_index": team_index,
                }
            )
            overall_pick += 1
    return pick_slots


def score_candidate_pool(
    candidate_pool: pd.DataFrame,
    team: TeamState,
    teams: List[TeamState],
    round_number: int,
    mode: str = "self",
) -> pd.DataFrame:
    scored = candidate_pool.copy()
    pool_reference_points = get_position_pool_reference_points(
        candidate_pool,
        teams,
        value_column="public_board_score" if "public_board_score" in candidate_pool.columns else "draft_value",
    )
    scored["need_multiplier"] = scored["position"].map(
        lambda pos: get_position_need_multiplier(team, pos, round_number)
    )
    scored["pool_reference_points"] = scored["position"].map(pool_reference_points)
    scored["pool_multiplier"] = 1.0 + (
        POOL_PRESSURE_WEIGHT
        * (
            (scored["public_board_score"] - scored["pool_reference_points"])
            .clip(lower=0.0)
            / scored["pool_reference_points"].clip(lower=1.0)
        )
    )
    scored["stack_multiplier"] = scored.apply(
        lambda row: get_stack_multiplier(team, row),
        axis=1,
        result_type="reduce",
    )
    scored["competition_multiplier"] = scored.apply(
        lambda row: get_competition_multiplier(team, row),
        axis=1,
        result_type="reduce",
    )
    scored["submodular_multiplier"] = scored.apply(
        lambda row: get_submodular_multiplier(team, row),
        axis=1,
        result_type="reduce",
    )
    scored["position_preference_multiplier"] = scored["position"].map(
        lambda pos: get_position_preference_multiplier(team, pos)
    )
    scored["robust_utility"] = (
        scored["starter_surplus_points"].clip(lower=0.0)
        + (0.42 * (scored["robust_expected_points"] - scored["replacement_projection"]).clip(lower=0.0))
        + (0.20 * (scored["ceiling_priority_points"] - scored["predicted_points"]).clip(lower=0.0))
        + (0.12 * scored["upside_tail_points"].clip(lower=0.0))
        - (0.12 * scored["downside_tail_points"].clip(lower=0.0))
    )

    if mode == "opponent":
        scored["pick_score"] = (
            (OPPONENT_PUBLIC_WEIGHT * scored["public_board_score"])
            + (OPPONENT_NEED_WEIGHT * scored["public_board_score"] * scored["need_multiplier"])
            + (0.20 * scored["robust_utility"])
        ) * scored["pool_multiplier"] * scored["position_preference_multiplier"] * scored["submodular_multiplier"]
        return scored

    scored["pick_score"] = (
        (0.58 * scored["draft_value"])
        + (0.42 * scored["robust_utility"])
    ) * (
        scored["need_multiplier"]
        * scored["pool_multiplier"]
        * scored["stack_multiplier"]
        * scored["competition_multiplier"]
        * scored["submodular_multiplier"]
    )
    scored["baseline_pick_score"] = (
        scored["draft_value"]
        * scored["need_multiplier"]
        * scored["pool_multiplier"]
        * scored["stack_multiplier"]
        * scored["competition_multiplier"]
    )
    return scored


def simulate_opponent_pick(
    available_df: pd.DataFrame,
    teams: List[TeamState],
    slot: dict,
) -> Tuple[pd.DataFrame, dict]:
    team = teams[slot["team_index"]]
    candidate_pool = available_df.loc[
        available_df.apply(lambda row: is_candidate_eligible(team, row, slot["round"]), axis=1)
    ].copy()
    if candidate_pool.empty:
        candidate_pool = available_df.copy()
    scored = score_candidate_pool(candidate_pool, team, teams, round_number=slot["round"], mode="opponent")
    top_candidates = scored.nlargest(min(TOP_CANDIDATES_TO_SAMPLE, len(scored)), "pick_score").reset_index()
    opponent_rng = np.random.default_rng(
        stable_seed(RANDOM_SEED, team.name, team.manager_profile, slot["overall_pick"], "opponent")
    )
    chosen_relative_idx = softmax_choice(top_candidates["pick_score"].to_numpy() / OPPONENT_TEMPERATURE, opponent_rng)
    chosen_index = int(top_candidates.loc[chosen_relative_idx, "index"])
    chosen_player = available_df.loc[chosen_index].to_dict()
    chosen_player["round"] = slot["round"]
    chosen_player["pick_in_round"] = slot["pick_in_round"]
    chosen_player["overall_pick"] = slot["overall_pick"]
    chosen_player["fantasy_team"] = team.name
    team.roster.append(chosen_player)
    updated_available = available_df.drop(index=chosen_index).reset_index(drop=True)
    return updated_available, chosen_player


def evaluate_rollout_score(
    candidate_row: pd.Series,
    team_index: int,
    teams: List[TeamState],
    available_df: pd.DataFrame,
    pick_slots: List[dict],
    current_slot_idx: int,
) -> float:
    immediate_score = float(candidate_row["pick_score"])
    next_own_slot_idx = None
    for future_slot_idx in range(current_slot_idx + 1, len(pick_slots)):
        if pick_slots[future_slot_idx]["team_index"] == team_index:
            next_own_slot_idx = future_slot_idx
            break
    if next_own_slot_idx is None:
        return immediate_score

    teams_sim = clone_teams(teams)
    chosen_player = candidate_row.to_dict()
    chosen_player["round"] = pick_slots[current_slot_idx]["round"]
    chosen_player["pick_in_round"] = pick_slots[current_slot_idx]["pick_in_round"]
    chosen_player["overall_pick"] = pick_slots[current_slot_idx]["overall_pick"]
    chosen_player["fantasy_team"] = teams_sim[team_index].name
    teams_sim[team_index].roster.append(chosen_player)
    remaining_available = available_df.drop(index=int(candidate_row.name)).reset_index(drop=True)

    for slot_idx in range(current_slot_idx + 1, next_own_slot_idx):
        remaining_available, _ = simulate_opponent_pick(
            remaining_available,
            teams_sim,
            pick_slots[slot_idx],
        )

    next_slot = pick_slots[next_own_slot_idx]
    next_team = teams_sim[team_index]
    next_candidate_pool = remaining_available.loc[
        remaining_available.apply(lambda row: is_candidate_eligible(next_team, row, next_slot["round"]), axis=1)
    ].copy()
    if next_candidate_pool.empty:
        next_candidate_pool = remaining_available.copy()
    next_scored_pool = score_candidate_pool(next_candidate_pool, next_team, teams_sim, next_slot["round"], mode="self")
    if next_scored_pool.empty:
        return immediate_score
    future_best_score = float(next_scored_pool["pick_score"].max())
    return immediate_score + (ROLLOUT_NEXT_PICK_WEIGHT * future_best_score)


def simulate_draft(
    projections_df: pd.DataFrame,
    league_size: int = LEAGUE_SIZE,
    rounds: int = DRAFT_ROUNDS,
    seed: int = RANDOM_SEED,
    add_randomness: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[TeamState]]:
    rng = np.random.default_rng(seed)
    teams = initialize_teams(league_size, rng)
    available = projections_df.sort_values(["draft_value", "predicted_points"], ascending=False).copy()
    available = available.reset_index(drop=True)
    draft_records: List[dict] = []
    pick_slots = build_pick_slots(league_size, rounds)

    for slot_idx, slot in enumerate(pick_slots):
        team_index = slot["team_index"]
        team = teams[team_index]
        candidate_pool = available.copy()
        eligibility_mask = candidate_pool.apply(
            lambda row: is_candidate_eligible(team, row, slot["round"]),
            axis=1,
        )
        if eligibility_mask.any():
            candidate_pool = candidate_pool.loc[eligibility_mask].copy()

        candidate_pool = score_candidate_pool(candidate_pool, team, teams, round_number=slot["round"], mode="self")
        rollout_pool = candidate_pool.nlargest(min(ROLLOUT_CANDIDATES, len(candidate_pool)), "pick_score").copy()
        rollout_pool["rollout_score"] = rollout_pool.apply(
            lambda row: evaluate_rollout_score(
                row,
                team_index=team_index,
                teams=teams,
                available_df=available,
                pick_slots=pick_slots,
                current_slot_idx=slot_idx,
            ),
            axis=1,
        )
        baseline_index = int(candidate_pool["pick_score"].idxmax())
        baseline_candidate = candidate_pool.loc[baseline_index]
        baseline_rollout_score = evaluate_rollout_score(
            baseline_candidate,
            team_index=team_index,
            teams=teams,
            available_df=available,
            pick_slots=pick_slots,
            current_slot_idx=slot_idx,
        )

        if add_randomness:
            top_candidates = rollout_pool.nlargest(TOP_CANDIDATES_TO_SAMPLE, "rollout_score").reset_index()
            standardized_rollout = top_candidates["rollout_score"].to_numpy()
            standardized_baseline = candidate_pool.nlargest(TOP_CANDIDATES_TO_SAMPLE, "pick_score").reset_index()
            chosen_relative_idx = softmax_choice(top_candidates["rollout_score"].to_numpy(), rng)
            chosen_index = int(top_candidates.loc[chosen_relative_idx, "index"])
            chosen_rollout_score = float(top_candidates.loc[chosen_relative_idx, "rollout_score"])
            target_probabilities = np.exp(
                (standardized_rollout - standardized_rollout.max())
                / max(standardized_rollout.std(ddof=0), 1.0)
            )
            target_probabilities = target_probabilities / target_probabilities.sum()
            behavior_logits = standardized_baseline["pick_score"].to_numpy()
            behavior_probabilities = np.exp(
                (behavior_logits - behavior_logits.max()) / max(behavior_logits.std(ddof=0), 1.0)
            )
            behavior_probabilities = behavior_probabilities / behavior_probabilities.sum()
            chosen_target_probability = float(target_probabilities[chosen_relative_idx])
            behavior_match = np.flatnonzero(standardized_baseline["index"].to_numpy() == chosen_index)
            chosen_behavior_probability = (
                float(behavior_probabilities[int(behavior_match[0])])
                if behavior_match.size > 0
                else float(1.0 / len(available))
            )
        else:
            chosen_index = int(rollout_pool["rollout_score"].idxmax())
            chosen_rollout_score = float(rollout_pool.loc[chosen_index, "rollout_score"])
            chosen_target_probability = 1.0
            chosen_behavior_probability = 1.0

        use_safe_fallback = chosen_rollout_score < (baseline_rollout_score - SAFE_POLICY_MARGIN)
        if use_safe_fallback:
            chosen_index = baseline_index
            chosen_rollout_score = float(baseline_rollout_score)
            chosen_target_probability = SAFE_POLICY_BLEND
            chosen_behavior_probability = 1.0

        chosen_player = available.loc[chosen_index].to_dict()
        chosen_player["round"] = slot["round"]
        chosen_player["pick_in_round"] = slot["pick_in_round"]
        chosen_player["overall_pick"] = slot["overall_pick"]
        chosen_player["fantasy_team"] = team.name
        chosen_player["manager_profile"] = team.manager_profile
        chosen_player["baseline_rollout_score"] = float(baseline_rollout_score)
        chosen_player["chosen_rollout_score"] = float(chosen_rollout_score)
        chosen_player["estimated_policy_gain"] = float(
            chosen_player["chosen_rollout_score"] - baseline_rollout_score
        )
        chosen_player["target_propensity"] = float(chosen_target_probability)
        chosen_player["behavior_propensity"] = float(chosen_behavior_probability)
        chosen_player["importance_weight"] = float(
            chosen_target_probability / max(chosen_behavior_probability, 1e-6)
        )
        chosen_player["used_safe_fallback"] = int(use_safe_fallback)
        team.roster.append(chosen_player)
        draft_records.append(chosen_player)

        available = available.drop(index=chosen_index).reset_index(drop=True)

    draft_board = pd.DataFrame(draft_records).sort_values("overall_pick").reset_index(drop=True)
    team_rosters = {
        team.name: draft_board[draft_board["fantasy_team"] == team.name]
        .sort_values("overall_pick")
        .reset_index(drop=True)
        for team in teams
    }
    return draft_board, team_rosters, teams


def select_optimal_lineup(roster_df: pd.DataFrame, point_column: str) -> pd.DataFrame:
    remaining = roster_df.sort_values(point_column, ascending=False).copy()
    selected_rows: List[dict] = []
    used_keys: set = set()

    def take_players(position: str, count: int, slot_prefix: str) -> None:
        nonlocal remaining
        candidates = remaining[(remaining["position"] == position) & (~remaining["player_key"].isin(used_keys))].head(count)
        for slot_index, (_, player) in enumerate(candidates.iterrows(), start=1):
            used_keys.add(player["player_key"])
            player_row = player.to_dict()
            player_row["slot"] = f"{slot_prefix}{slot_index}" if count > 1 else slot_prefix
            selected_rows.append(player_row)

    take_players("QB", 1, "QB")
    take_players("RB", 2, "RB")
    take_players("WR", 2, "WR")
    take_players("TE", 1, "TE")

    flex_candidates = remaining[
        remaining["position"].isin(FLEX_POSITIONS) & (~remaining["player_key"].isin(used_keys))
    ].head(FLEX_SLOTS)
    for _, player in flex_candidates.iterrows():
        used_keys.add(player["player_key"])
        player_row = player.to_dict()
        player_row["slot"] = "FLEX"
        selected_rows.append(player_row)

    lineup = pd.DataFrame(selected_rows)
    if lineup.empty:
        return lineup

    slot_order = {"QB": 0, "RB1": 1, "RB2": 2, "WR1": 3, "WR2": 4, "TE": 5, "FLEX": 6}
    lineup["slot_order"] = lineup["slot"].map(slot_order).fillna(99)
    lineup = lineup.sort_values("slot_order").drop(columns="slot_order").reset_index(drop=True)
    return lineup


def get_player_week_count(player: pd.Series, point_column: str) -> int:
    raw_games = player.get("predicted_games") if point_column == "predicted_points" else player.get("games_played")
    if raw_games is None or (isinstance(raw_games, float) and np.isnan(raw_games)):
        fallback = player.get("games_played", player.get("predicted_games", WEEKLY_SIM_WEEKS))
        raw_games = fallback
    try:
        games = int(np.clip(np.rint(float(raw_games)), 0, WEEKLY_SIM_WEEKS))
    except (TypeError, ValueError):
        games = WEEKLY_SIM_WEEKS
    return games


def simulate_player_weekly_points(
    player: pd.Series,
    point_column: str,
    seed: int = RANDOM_SEED,
    weeks: int = WEEKLY_SIM_WEEKS,
) -> np.ndarray:
    try:
        total_points = float(player.get(point_column, 0.0))
    except (TypeError, ValueError):
        total_points = 0.0

    if abs(total_points) < 1e-9:
        return np.zeros(weeks)

    active_weeks = get_player_week_count(player, point_column)
    if active_weeks <= 0:
        return np.zeros(weeks)

    position = str(player.get("position", "WR"))
    alpha = WEEKLY_DIRICHLET_ALPHA.get(position, 12.0)
    sigma = WEEKLY_LOGNORMAL_SIGMA.get(position, 0.10)
    player_key = player.get("player_key") or player.get("player_name", "unknown")
    season = int(player.get("season", 2024))
    rng = np.random.default_rng(stable_seed(seed, season, player_key, point_column, weeks))

    week_points = np.zeros(weeks)
    week_indices = np.arange(weeks)
    if active_weeks < weeks:
        week_indices = np.sort(rng.choice(weeks, size=active_weeks, replace=False))

    base_weights = rng.dirichlet(np.full(active_weeks, alpha, dtype=float))
    noise = rng.lognormal(mean=-(sigma**2) / 2.0, sigma=sigma, size=active_weeks)
    adjusted_weights = base_weights * noise
    adjusted_weights = adjusted_weights / adjusted_weights.sum()
    week_points[week_indices] = adjusted_weights * abs(total_points)
    return week_points * np.sign(total_points)


def simulate_team_weekly_lineups(
    team_name: str,
    roster_df: pd.DataFrame,
    point_column: str,
    seed: int = RANDOM_SEED,
    record_lineups: bool = False,
) -> Tuple[float, pd.DataFrame]:
    if roster_df.empty:
        empty_columns = [
            "week",
            "slot",
            "fantasy_team",
            "player_name",
            "team",
            "position",
            "round",
            "overall_pick",
            "predicted_points",
            "actual_points",
            "weekly_points",
            "team_name",
        ]
        return 0.0, pd.DataFrame(columns=empty_columns)

    weekly_matrix = np.vstack(
        [simulate_player_weekly_points(player, point_column=point_column, seed=seed) for _, player in roster_df.iterrows()]
    )
    week_totals: List[float] = []
    lineup_frames: List[pd.DataFrame] = []

    for week_index in range(WEEKLY_SIM_WEEKS):
        weekly_roster = roster_df.copy()
        weekly_roster["week_points"] = weekly_matrix[:, week_index]
        lineup_df = select_optimal_lineup(weekly_roster, point_column="week_points")
        week_totals.append(float(lineup_df["week_points"].sum()) if not lineup_df.empty else 0.0)

        if record_lineups and not lineup_df.empty:
            weekly_lineup = lineup_df[
                [
                    "slot",
                    "fantasy_team",
                    "player_name",
                    "team",
                    "position",
                    "round",
                    "overall_pick",
                    "predicted_points",
                    "actual_points",
                    "week_points",
                ]
            ].copy()
            weekly_lineup["week"] = week_index + 1
            weekly_lineup["weekly_points"] = weekly_lineup["week_points"]
            weekly_lineup["team_name"] = team_name
            lineup_frames.append(weekly_lineup.drop(columns="week_points"))

    lineups = (
        pd.concat(lineup_frames, ignore_index=True)
        if lineup_frames
        else pd.DataFrame(
            columns=[
                "week",
                "slot",
                "fantasy_team",
                "player_name",
                "team",
                "position",
                "round",
                "overall_pick",
                "predicted_points",
                "actual_points",
                "weekly_points",
                "team_name",
            ]
        )
    )
    if not lineups.empty:
        lineups = lineups[
            [
                "week",
                "slot",
                "fantasy_team",
                "player_name",
                "team",
                "position",
                "round",
                "overall_pick",
                "predicted_points",
                "actual_points",
                "weekly_points",
                "team_name",
            ]
        ]
    return float(np.sum(week_totals)), lineups


def simulate_season(team_rosters: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    team_result_rows: List[dict] = []
    lineup_frames: List[pd.DataFrame] = []

    for team_name, roster_df in team_rosters.items():
        actual_total, lineup_df = simulate_team_weekly_lineups(
            team_name=team_name,
            roster_df=roster_df,
            point_column="actual_points",
            seed=RANDOM_SEED,
            record_lineups=True,
        )
        projected_total, _ = simulate_team_weekly_lineups(
            team_name=team_name,
            roster_df=roster_df,
            point_column="predicted_points",
            seed=RANDOM_SEED + 1,
            record_lineups=False,
        )

        team_result_rows.append(
            {
                "team_name": team_name,
                "projected_starting_points": projected_total,
                "total_season_points": actual_total,
            }
        )

        if not lineup_df.empty:
            enriched_lineup = lineup_df[
                [
                    "week",
                    "slot",
                    "fantasy_team",
                    "player_name",
                    "team",
                    "position",
                    "round",
                    "overall_pick",
                    "predicted_points",
                    "actual_points",
                    "weekly_points",
                ]
            ].copy()
            enriched_lineup["team_name"] = team_name
            lineup_frames.append(enriched_lineup)

    team_results = pd.DataFrame(team_result_rows).sort_values(
        ["total_season_points", "projected_starting_points"], ascending=False
    )
    team_results["rank"] = np.arange(1, len(team_results) + 1)
    team_results = team_results[["rank", "team_name", "total_season_points", "projected_starting_points"]]

    lineups = (
        pd.concat(lineup_frames, ignore_index=True)
        if lineup_frames
        else pd.DataFrame(
            columns=[
                "week",
                "slot",
                "fantasy_team",
                "player_name",
                "team",
                "position",
                "round",
                "overall_pick",
                "predicted_points",
                "actual_points",
                "weekly_points",
                "team_name",
            ]
        )
    )
    if not lineups.empty:
        lineups["slot_order"] = lineups["slot"].map(
            {"QB": 0, "RB1": 1, "RB2": 2, "WR1": 3, "WR2": 4, "TE": 5, "FLEX": 6}
        ).fillna(99)
        lineups = (
            lineups.sort_values(["team_name", "week", "slot_order", "overall_pick"])
            .drop(columns="slot_order")
            .reset_index(drop=True)
        )
    return team_results, lineups


def build_top20_table(projections_df: pd.DataFrame) -> pd.DataFrame:
    data = projections_df.copy()
    data["predicted_rank"] = data["predicted_points"].rank(method="min", ascending=False)
    data["actual_rank"] = data["actual_points"].rank(method="min", ascending=False)
    top20 = data.sort_values("draft_value", ascending=False).head(20).copy()
    top20["point_error"] = top20["predicted_points"] - top20["actual_points"]
    return top20[
        [
            "predicted_rank",
            "actual_rank",
            "player_name",
            "team",
            "position",
            "predicted_points",
            "actual_points",
            "predicted_vor",
            "actual_vor",
            "draft_value",
            "point_error",
        ]
    ].sort_values("predicted_rank")


def save_outputs(
    output_dir: Path,
    comparison_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    draft_board_df: pd.DataFrame,
    team_results_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    team_rosters: Dict[str, pd.DataFrame],
    top20_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    validation_df.to_csv(output_dir / "validation_predictions_2022_2023.csv", index=False)
    projections_df.sort_values("draft_value", ascending=False).to_csv(output_dir / "player_projections_2024.csv", index=False)
    draft_board_df.to_csv(output_dir / "draft_board.csv", index=False)
    team_results_df.to_csv(output_dir / "team_results.csv", index=False)
    lineups_df.to_csv(output_dir / "starting_lineups.csv", index=False)
    top20_df.to_csv(output_dir / "top20_predicted_vs_actual.csv", index=False)

    roster_frames = []
    for team_name, roster_df in team_rosters.items():
        roster_copy = roster_df.copy()
        roster_copy["team_name"] = team_name
        roster_frames.append(roster_copy)
        safe_name = team_name.lower().replace(" ", "_")
        roster_copy.to_csv(output_dir / f"{safe_name}_drafted_players.csv", index=False)

    if roster_frames:
        pd.concat(roster_frames, ignore_index=True).to_csv(output_dir / "all_team_rosters.csv", index=False)


def dataframe_to_text(title: str, df: pd.DataFrame) -> str:
    if df.empty:
        return f"{title}\n<empty>\n"
    return f"{title}\n{df.to_string(index=False)}\n"


def write_report(
    output_dir: Path,
    comparison_df: pd.DataFrame,
    draft_board_df: pd.DataFrame,
    team_results_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    team_rosters: Dict[str, pd.DataFrame],
    top20_df: pd.DataFrame,
) -> None:
    sections = [
        dataframe_to_text("Model Comparison", comparison_df.round(3)),
        dataframe_to_text(
            "Team Results",
            team_results_df.round({"total_season_points": 2, "projected_starting_points": 2}),
        ),
        dataframe_to_text(
            "Draft Board",
            draft_board_df[
                [
                    "overall_pick",
                    "round",
                    "fantasy_team",
                    "player_name",
                    "team",
                    "position",
                    "predicted_points",
                    "actual_points",
                    "draft_value",
                ]
            ].round({"predicted_points": 2, "actual_points": 2, "draft_value": 2}),
        ),
        dataframe_to_text(
            "Top 20 Predicted Value vs Actual Outcome",
            top20_df.round(
                {
                    "predicted_points": 2,
                    "actual_points": 2,
                    "predicted_vor": 2,
                    "actual_vor": 2,
                    "draft_value": 2,
                    "point_error": 2,
                }
            ),
        ),
    ]

    for team_name, roster_df in team_rosters.items():
        roster_view = roster_df[
            [
                "round",
                "overall_pick",
                "player_name",
                "team",
                "position",
                "predicted_points",
                "actual_points",
                "draft_value",
            ]
        ].round({"predicted_points": 2, "actual_points": 2, "draft_value": 2})
        sections.append(dataframe_to_text(f"{team_name} Drafted Players", roster_view))

        lineup_columns = [
            "week",
            "slot",
            "player_name",
            "team",
            "position",
            "round",
            "overall_pick",
            "weekly_points",
            "predicted_points",
            "actual_points",
        ]
        lineup_view = lineups_df[lineups_df["team_name"] == team_name][lineup_columns].round(
            {"weekly_points": 2, "predicted_points": 2, "actual_points": 2}
        )
        sections.append(dataframe_to_text(f"{team_name} Weekly Starting Lineups", lineup_view))

    (output_dir / "simulation_report.txt").write_text("\n".join(sections))


def print_summary(
    comparison_df: pd.DataFrame,
    team_results_df: pd.DataFrame,
    top20_df: pd.DataFrame,
) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 200)

    print("\nMODEL COMPARISON")
    print(comparison_df.round(3).to_string(index=False))

    print("\nFINAL TEAM RESULTS")
    print(team_results_df.round({"total_season_points": 2, "projected_starting_points": 2}).to_string(index=False))

    print("\nTOP 20 PREDICTED VALUE VS ACTUAL OUTCOME")
    print(
        top20_df.round(
            {
                "predicted_points": 2,
                "actual_points": 2,
                "predicted_vor": 2,
                "actual_vor": 2,
                "draft_value": 2,
                "point_error": 2,
            }
        ).to_string(index=False)
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    csv_path = root / "fantasy_data.csv"
    output_dir = root / "outputs"

    print("Loading data...")
    df = load_data(csv_path)
    print("Building features and historical VOR targets...")
    feature_df = build_features(df)
    print("Training models on pre-2024 seasons and validating on 2022-2023...")
    model_bundle, comparison_df, validation_df = train_model(feature_df)
    print(f"Selected projection blends by position: {model_bundle['selected_model_summary']}")
    print("Projecting 2024 player values...")
    projections_df = project_2024_players(feature_df, model_bundle)
    print("Simulating the 8-team snake draft...")
    draft_board_df, team_rosters, _ = simulate_draft(projections_df, add_randomness=False)
    print("Simulating the 2024 season with weekly point splits and best-ball lineup optimization...")
    team_results_df, lineups_df = simulate_season(team_rosters)
    top20_df = build_top20_table(projections_df)

    print("Saving reports and CSV outputs...")
    save_outputs(
        output_dir=output_dir,
        comparison_df=comparison_df,
        validation_df=validation_df,
        projections_df=projections_df,
        draft_board_df=draft_board_df,
        team_results_df=team_results_df,
        lineups_df=lineups_df,
        team_rosters=team_rosters,
        top20_df=top20_df,
    )
    write_report(
        output_dir=output_dir,
        comparison_df=comparison_df,
        draft_board_df=draft_board_df,
        team_results_df=team_results_df,
        lineups_df=lineups_df,
        team_rosters=team_rosters,
        top20_df=top20_df,
    )
    print("Complete.")
    print_summary(comparison_df, team_results_df, top20_df)


if __name__ == "__main__":
    main()
