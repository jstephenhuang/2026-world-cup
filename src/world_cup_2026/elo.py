"""Elo ratings, computed by replaying match history chronologically.

After `compute_elo(df)` runs, the returned `ratings` dict holds each team's
CURRENT strength (its rating after the last match in the data). That dict is
what we feed the model to predict 2026 fixtures.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd

START = 1500.0
K = 30.0
HOME_ADV = 65.0  # rating points added to the home side when not on neutral ground


def _expected(a: float, b: float) -> float:
    """Expected score for a vs b (logistic Elo curve)."""
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def compute_elo(df: pd.DataFrame):
    """Walk matches in date order, attaching pre-match Elo to each row.

    Returns (df_with_elo_columns, current_ratings_dict).
    df must be sorted by date and contain:
        home_team, away_team, home_score, away_score, neutral
    """
    ratings: dict[str, float] = defaultdict(lambda: START)
    home_elo, away_elo = [], []

    for r in df.itertuples():
        rh, ra = ratings[r.home_team], ratings[r.away_team]
        # record the PRE-match rating (this is what the model is allowed to see)
        home_elo.append(rh)
        away_elo.append(ra)

        adv = HOME_ADV if not r.neutral else 0.0
        exp_h = _expected(rh + adv, ra)

        if r.home_score > r.away_score:
            res_h = 1.0
        elif r.home_score == r.away_score:
            res_h = 0.5
        else:
            res_h = 0.0

        # bigger margins move ratings more
        margin_mult = 1.0 + np.log1p(abs(r.home_score - r.away_score))
        delta = K * margin_mult * (res_h - exp_h)

        ratings[r.home_team] = rh + delta
        ratings[r.away_team] = ra - delta

    df = df.copy()
    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    return df, dict(ratings)
