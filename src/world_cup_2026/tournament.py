"""Monte Carlo simulation of the 2026 World Cup.

Format (FIFA, confirmed): 48 teams in 12 groups of four. Each team plays the
other three in its group. The 12 group winners, 12 runners-up, and the 8 best
third-placed teams (32 total) advance to a Round of 32, then R16, QF, SF, final
— single elimination.

Group standings tiebreakers used here (simplified from FIFA's full list):
    1. points  2. goal difference  3. goals scored  4. random draw
We track goals scored/conceded per group game so third-place ranking works.

Knockout draws in real life follow a fixed bracket template tied to which
groups the third-place teams come from. That template is genuinely fiddly;
this module uses a reseeding approach (rank all 32 qualifiers by group
performance and pair strongest vs weakest each round). It's a reasonable,
unbiased approximation for picking. If you want the exact official bracket,
replace `build_round_of_32` with the hardcoded slot map.
"""
from __future__ import annotations
from collections import defaultdict
import itertools
import random
import numpy as np
import pandas as pd

from .data import FEATURES


class Predictor:
    """Wraps the trained model + current Elo + last-known form to score any
    hypothetical 2026 fixture."""

    def __init__(self, clf, ratings: dict, form: dict, league_avg: float):
        self.clf = clf
        self.ratings = ratings
        self.form = form          # team -> dict(gf, ga)
        self.league_avg = league_avg
        self._cache: dict = {}    # (home, away, neutral) -> (p_away,p_draw,p_home)

    def _row(self, home: str, away: str, neutral: int = 1) -> pd.DataFrame:
        rh = self.ratings.get(home, 1500.0)
        ra = self.ratings.get(away, 1500.0)
        fh = self.form.get(home, {"gf": self.league_avg, "ga": self.league_avg})
        fa = self.form.get(away, {"gf": self.league_avg, "ga": self.league_avg})
        return pd.DataFrame([{
            "elo_diff": rh - ra,
            "home_elo": rh,
            "away_elo": ra,
            "neutral": neutral,
            "is_tournament": 1,
            "home_form_gf": fh["gf"],
            "home_form_ga": fh["ga"],
            "away_form_gf": fa["gf"],
            "away_form_ga": fa["ga"],
        }])[FEATURES]

    def probs(self, home: str, away: str, neutral: int = 1):
        """Return (p_away, p_draw, p_home). Memoized: each unique matchup hits
        the model only once, so Monte Carlo reuses results across iterations."""
        key = (home, away, neutral)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        p = self.clf.predict_proba(self._row(home, away, neutral))[0]
        # classes are [0,1,2] = [away, draw, home]
        result = (float(p[0]), float(p[1]), float(p[2]))
        self._cache[key] = result
        return result


def play_group_match(pred: Predictor, h: str, a: str):
    """Sample a group result. Returns (home_pts, away_pts, home_gf, away_gf).
    Goals are sampled lightly just to drive tiebreakers."""
    p_away, p_draw, p_home = pred.probs(h, a)
    r = random.random()
    if r < p_home:
        hg, ag = _sample_score(win=True)
        return 3, 0, hg, ag
    elif r < p_home + p_draw:
        g = random.choice([0, 1, 2])
        return 1, 1, g, g
    else:
        ag, hg = _sample_score(win=True)
        return 0, 3, hg, ag


def _sample_score(win: bool):
    """Cheap scoreline: winner 1-3 goals, loser strictly fewer."""
    w = random.choice([1, 1, 2, 2, 2, 3])
    l = random.randint(0, w - 1)
    return w, l


def simulate_group(pred: Predictor, teams: list[str]):
    pts = defaultdict(int)
    gf = defaultdict(int)
    ga = defaultdict(int)
    for h, a in itertools.combinations(teams, 2):
        hp, ap, hg, ag = play_group_match(pred, h, a)
        pts[h] += hp; pts[a] += ap
        gf[h] += hg; ga[h] += ag
        gf[a] += ag; ga[a] += hg

    def key(t):
        return (pts[t], gf[t] - ga[t], gf[t], random.random())

    ranked = sorted(teams, key=key, reverse=True)
    standings = [
        {"team": t, "pts": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]}
        for t in ranked
    ]
    return ranked, standings


def simulate_knockout_match(pred: Predictor, t1: str, t2: str) -> str:
    p_away, p_draw, p_home = pred.probs(t1, t2)
    # split the draw mass ~evenly (penalty shootouts are close to a coin flip)
    p_t1 = p_home + p_draw * 0.5
    total = p_t1 + (p_away + p_draw * 0.5)
    return t1 if random.random() < p_t1 / total else t2


def _seed_rank(qualifiers):
    """Order qualifiers strongest->weakest for reseeded bracket pairing."""
    def k(q):
        return (q["pts"], q["gd"], q["gf"], random.random())
    return sorted(qualifiers, key=k, reverse=True)


def simulate_tournament(pred: Predictor, groups: dict[str, list[str]]):
    """Run one full tournament. Returns dict team -> furthest stage reached."""
    reached = {}
    winners, runners, thirds = [], [], []

    for gname, teams in groups.items():
        ranked, standings = simulate_group(pred, teams)
        for t in teams:
            reached[t] = "group"
        winners.append(standings[0])
        runners.append(standings[1])
        thirds.append(standings[2])

    # 8 best third-placed teams
    best_thirds = _seed_rank(thirds)[:8]
    qualifiers = winners + runners + best_thirds  # 32 teams
    for q in qualifiers:
        reached[q["team"]] = "R32"

    # reseeded single elimination
    field = _seed_rank(qualifiers)
    stage_names = ["R32", "R16", "QF", "SF", "F"]
    next_stage = {"R32": "R16", "R16": "QF", "QF": "SF", "SF": "F", "F": "champion"}

    current = [q["team"] for q in field]
    stage_idx = 0
    while len(current) > 1:
        # pair strongest vs weakest
        pairs = [(current[i], current[len(current) - 1 - i])
                 for i in range(len(current) // 2)]
        survivors = []
        for t1, t2 in pairs:
            w = simulate_knockout_match(pred, t1, t2)
            survivors.append(w)
            reached[w] = next_stage[stage_names[stage_idx]]
        current = survivors
        stage_idx += 1

    reached[current[0]] = "champion"
    return reached


def monte_carlo(pred: Predictor, groups: dict[str, list[str]], n: int = 20000):
    stage_order = ["group", "R32", "R16", "QF", "SF", "F", "champion"]
    rank = {s: i for i, s in enumerate(stage_order)}
    tally = defaultdict(lambda: defaultdict(int))  # team -> stage -> count

    for _ in range(n):
        reached = simulate_tournament(pred, groups)
        for team, stage in reached.items():
            # count "reached at least this far" for every stage up to max
            for s in stage_order[: rank[stage] + 1]:
                tally[team][s] += 1

    rows = []
    for team, d in tally.items():
        rows.append({
            "team": team,
            "win_title": d["champion"] / n,
            "reach_final": d["F"] / n,
            "reach_semi": d["SF"] / n,
            "reach_quarter": d["QF"] / n,
            "reach_r16": d["R16"] / n,
            "advance_group": d["R32"] / n,
        })
    out = pd.DataFrame(rows).sort_values("win_title", ascending=False)
    return out.reset_index(drop=True)
