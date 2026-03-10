"""
utils/bracket.py
-----------------
Simulates the full 64-team NCAA Tournament bracket using
win probabilities drawn from the MCMC posterior.

Bracket structure
-----------------
  Round 1 : 64 → 32  (Round of 64)
  Round 2 : 32 → 16  (Round of 32)
  Round 3 : 16 → 8   (Sweet 16)
  Round 4 : 8  → 4   (Elite 8)
  Round 5 : 4  → 2   (Final Four)
  Round 6 : 2  → 1   (Championship)
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from typing import Optional


ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


class BracketSimulator:
    """
    Simulates the NCAA Tournament bracket using posterior alpha samples.

    Parameters
    ----------
    teams_df      : DataFrame with [team, seed, region]
    alpha_samples : np.ndarray of shape (n_samples, n_teams)
    team_names    : list of team names aligned with alpha_samples columns
    seed          : random seed
    """

    def __init__(
        self,
        teams_df:      pd.DataFrame,
        alpha_samples: np.ndarray,
        team_names:    list[str],
        seed:          int = 42,
    ):
        self.teams_df      = teams_df.reset_index(drop=True)
        self.alpha_samples = alpha_samples
        self.team_names    = team_names
        self.idx           = {t: i for i, t in enumerate(team_names)}
        self.rng           = np.random.default_rng(seed)
        self.n_samples     = alpha_samples.shape[0]
        self._build_bracket()

    # ── bracket construction ──────────────────────────────────────────────

    def _build_bracket(self):
        """
        Build the standard 64-team bracket.
        Seeds are paired: (1 vs 16, 8 vs 9, 5 vs 12, 4 vs 13,
                           6 vs 11, 3 vs 14, 7 vs 10, 2 vs 15)
        """
        seed_pairings = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
        regions       = ["East","West","South","Midwest"]

        self.bracket: dict[str, list] = {}   # region → ordered list of 16 teams
        for region in regions:
            region_teams = self.teams_df[self.teams_df["region"] == region].copy()
            slot = []
            for s1, s2 in seed_pairings:
                t1 = region_teams[region_teams["seed"] == s1]
                t2 = region_teams[region_teams["seed"] == s2]
                t1_name = t1["team"].iloc[0] if len(t1) > 0 else f"Unknown({region},{s1})"
                t2_name = t2["team"].iloc[0] if len(t2) > 0 else f"Unknown({region},{s2})"
                slot.extend([t1_name, t2_name])
            self.bracket[region] = slot

    # ── single tournament simulation ─────────────────────────────────────

    def _simulate_game(self, team_a: str, team_b: str, alpha: np.ndarray) -> str:
        """Simulate one game given a single alpha draw."""
        ia = self.idx.get(team_a, -1)
        ib = self.idx.get(team_b, -1)
        if ia == -1 or ib == -1:
            # Unknown team: 50/50
            return team_a if self.rng.random() < 0.5 else team_b
        diff = alpha[ia] - alpha[ib]
        p_a  = float(expit(diff))
        return team_a if self.rng.random() < p_a else team_b

    def _simulate_single_region(self, teams: list[str], alpha: np.ndarray) -> dict:
        """Simulate one regional bracket, return champion and per-round results."""
        current = list(teams)
        rounds  = {}
        for rnd in range(1, 5):   # rounds 1–4 within region
            next_round = []
            matchups   = []
            for i in range(0, len(current), 2):
                a, b = current[i], current[i+1]
                w = self._simulate_game(a, b, alpha)
                next_round.append(w)
                matchups.append({"team1": a, "team2": b, "winner": w, "round": rnd})
            rounds[rnd] = matchups
            current = next_round
        return {"champion": current[0], "rounds": rounds}

    def _simulate_tournament(self, alpha: np.ndarray) -> dict:
        """Full 64-team simulation using one posterior draw."""
        regions = ["East","West","South","Midwest"]
        regional_results = {}
        final_four = []

        for region in regions:
            res = self._simulate_single_region(self.bracket[region], alpha)
            regional_results[region] = res
            final_four.append(res["champion"])

        # Final Four: East vs West, South vs Midwest
        semifinal_1 = self._simulate_game(final_four[0], final_four[1], alpha)
        semifinal_2 = self._simulate_game(final_four[2], final_four[3], alpha)
        champion    = self._simulate_game(semifinal_1, semifinal_2, alpha)

        return {
            "regional": regional_results,
            "final_four": final_four,
            "semifinal_1": semifinal_1,
            "semifinal_2": semifinal_2,
            "champion": champion,
        }

    # ── Monte Carlo tournament simulation ────────────────────────────────

    def run_simulations(self, n_simulations: int = 10_000, verbose: bool = True) -> dict:
        """
        Run N full tournament simulations, sampling from posterior each time.

        Returns
        -------
        dict with keys:
          'championship_probs' : pd.Series  team → P(win championship)
          'final_four_probs'   : pd.Series  team → P(reach Final Four)
          'elite_eight_probs'  : pd.Series  team → P(reach Elite Eight)
          'sweet_16_probs'     : pd.Series  team → P(reach Sweet 16)
          'round_32_probs'     : pd.Series  team → P(reach Round of 32)
          'win_counts'         : dict
          'raw_results'        : list of simulation results
        """
        if verbose:
            print(f"[BracketSimulator] Running {n_simulations:,} tournament simulations...")

        counters = {
            "champion":   {t: 0 for t in self.team_names},
            "final_four": {t: 0 for t in self.team_names},
            "elite_8":    {t: 0 for t in self.team_names},
            "sweet_16":   {t: 0 for t in self.team_names},
            "round_32":   {t: 0 for t in self.team_names},
        }
        raw_results = []

        for sim in range(n_simulations):
            # Draw a posterior sample (with replacement)
            draw_idx = self.rng.integers(0, self.n_samples)
            alpha    = self.alpha_samples[draw_idx]

            result = self._simulate_tournament(alpha)
            raw_results.append(result)

            # Tally champion
            c = result["champion"]
            if c in counters["champion"]:
                counters["champion"][c] += 1

            # Tally Final Four
            for t in result["final_four"]:
                if t in counters["final_four"]:
                    counters["final_four"][t] += 1

            # Tally regional rounds
            for region, res in result["regional"].items():
                for rnd, matchups in res["rounds"].items():
                    for m in matchups:
                        w = m["winner"]
                        if rnd == 1 and w in counters["round_32"]:
                            counters["round_32"][w] += 1
                        elif rnd == 2 and w in counters["sweet_16"]:
                            counters["sweet_16"][w] += 1
                        elif rnd == 3 and w in counters["elite_8"]:
                            counters["elite_8"][w] += 1

            if verbose and (sim + 1) % 2000 == 0:
                print(f"  Completed {sim+1:,}/{n_simulations:,} simulations")

        def to_prob(counter):
            return pd.Series({t: v / n_simulations for t, v in counter.items()})

        if verbose:
            print("[BracketSimulator] Simulations complete.")

        return {
            "championship_probs": to_prob(counters["champion"]).sort_values(ascending=False),
            "final_four_probs":   to_prob(counters["final_four"]).sort_values(ascending=False),
            "elite_eight_probs":  to_prob(counters["elite_8"]).sort_values(ascending=False),
            "sweet_16_probs":     to_prob(counters["sweet_16"]).sort_values(ascending=False),
            "round_32_probs":     to_prob(counters["round_32"]).sort_values(ascending=False),
            "win_counts":         counters,
            "raw_results":        raw_results,
            "n_simulations":      n_simulations,
        }
