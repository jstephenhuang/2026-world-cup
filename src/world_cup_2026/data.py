"""Load the Kaggle results.csv and turn raw matches into training examples.

Kaggle dataset: "International football results from 1872 to 2026"
(file: results.csv) — https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

The hard rule throughout this module: a match's features may only use
information available BEFORE that match kicked off. Rolling stats are shifted
by one so a row never sees its own outcome (no leakage).
"""
from __future__ import annotations
from collections import defaultdict, deque
import numpy as np
import pandas as pd

FEATURES = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "neutral",
    "is_tournament",
    "home_form_gf",
    "home_form_ga",
    "away_form_gf",
    "away_form_ga",
]


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    needed = {"date", "home_team", "away_team", "home_score", "away_score",
              "tournament", "neutral"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"results.csv is missing columns: {missing}")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def label_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Target: 0 = away win, 1 = draw, 2 = home win."""
    df = df.copy()
    cond = [
        df.home_score > df.away_score,
        df.home_score == df.away_score,
    ]
    df["y"] = np.select(cond, [2, 1], default=0)
    return df


def add_rolling_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Average goals scored / conceded over each team's previous `n` matches.

    Implemented with a per-team rolling deque while walking forward in time,
    so each row only ever reflects matches that already happened.
    """
    df = df.copy()
    gf_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=n))
    ga_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=n))

    h_gf, h_ga, a_gf, a_ga = [], [], [], []

    def avg(d: deque, default: float) -> float:
        return sum(d) / len(d) if d else default

    league_avg = (df.home_score.mean() + df.away_score.mean()) / 2

    for r in df.itertuples():
        # features = state BEFORE this match
        h_gf.append(avg(gf_hist[r.home_team], league_avg))
        h_ga.append(avg(ga_hist[r.home_team], league_avg))
        a_gf.append(avg(gf_hist[r.away_team], league_avg))
        a_ga.append(avg(ga_hist[r.away_team], league_avg))

        # then update history with this match's result
        gf_hist[r.home_team].append(r.home_score)
        ga_hist[r.home_team].append(r.away_score)
        gf_hist[r.away_team].append(r.away_score)
        ga_hist[r.away_team].append(r.home_score)

    df["home_form_gf"] = h_gf
    df["home_form_ga"] = h_ga
    df["away_form_gf"] = a_gf
    df["away_form_ga"] = a_ga
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["neutral"] = df["neutral"].astype(int)
    df["is_tournament"] = (df["tournament"].str.lower() != "friendly").astype(int)
    return df
