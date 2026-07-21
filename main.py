"""End-to-end: load data -> Elo -> features -> train -> simulate 2026.

Usage (from the project root):
    uv run wc2026 --data path/to/results.csv
    uv run wc2026 --data results.csv --model xgb --sims 50000
    uv run python -m wc2026.main --data results.csv

Get results.csv from Kaggle:
    "International football results from 1872 to 2026"
    https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import sys

from src.world_cup_2026.data import (load_results, label_outcome, add_rolling_form,
                   build_features)
from src.world_cup_2026.elo import compute_elo
from src.world_cup_2026.model import train
from src.world_cup_2026.tournament import Predictor, monte_carlo
from src.world_cup_2026.groups import GROUPS_2026


def latest_form(df, n: int = 5):
    """Each team's average GF/GA over its most recent n matches in the data.
    Used as the 'current' form fed into 2026 predictions."""
    gf_hist = defaultdict(lambda: deque(maxlen=n))
    ga_hist = defaultdict(lambda: deque(maxlen=n))
    for r in df.itertuples():
        gf_hist[r.home_team].append(r.home_score)
        ga_hist[r.home_team].append(r.away_score)
        gf_hist[r.away_team].append(r.away_score)
        ga_hist[r.away_team].append(r.home_score)
    league_avg = (df.home_score.mean() + df.away_score.mean()) / 2
    form = {}
    for team in set(gf_hist) | set(ga_hist):
        gf = gf_hist[team]; ga = ga_hist[team]
        form[team] = {
            "gf": sum(gf) / len(gf) if gf else league_avg,
            "ga": sum(ga) / len(ga) if ga else league_avg,
        }
    return form, league_avg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Predict the 2026 World Cup.")
    p.add_argument("--data", required=True, help="path to Kaggle results.csv")
    p.add_argument("--model", choices=["rf", "xgb"], default="rf")
    p.add_argument("--sims", type=int, default=20000,
                   help="Monte Carlo iterations")
    p.add_argument("--split-date", default="2024-01-01",
                   help="train/test cutoff for evaluation")
    p.add_argument("--top", type=int, default=20,
                   help="how many teams to print")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Loading {args.data} ...")
    df = load_results(args.data)
    print(f"  {len(df):,} matches, {df.date.min().date()} -> {df.date.max().date()}")

    print("Computing Elo (replaying history) ...")
    df, ratings = compute_elo(df)

    print("Building features + labels ...")
    df = label_outcome(df)
    df = add_rolling_form(df, n=5)
    df = build_features(df)

    print(f"Training model ({args.model}) ...")
    clf = train(df, model=args.model, split_date=args.split_date)

    print("Snapshotting current form ...")
    form, league_avg = latest_form(df, n=5)

    pred = Predictor(clf, ratings, form, league_avg)

    # sanity: warn about teams in the groups that aren't in the data
    known = set(ratings)
    for g, teams in GROUPS_2026.items():
        for t in teams:
            if t not in known:
                print(f"  WARNING: '{t}' (group {g}) not found in data -> "
                      f"defaulting to Elo 1500. Fix the spelling in groups.py.",
                      file=sys.stderr)

    print(f"Simulating the tournament {args.sims:,} times ...")
    table = monte_carlo(pred, GROUPS_2026, n=args.sims)

    pd_opts = ("display.max_rows", None, "display.width", 120,
               "display.float_format", "{:.1%}".format)
    import pandas as pd
    with pd.option_context(*pd_opts):
        print("\n=== 2026 World Cup probabilities ===")
        print(table.head(args.top).to_string(index=False))

    table.to_csv("predictions_2026.csv", index=False)
    print("\nSaved full table -> predictions_2026.csv")


if __name__ == "__main__":
    main()
