"""
models/mcmc.py
--------------
Markov Chain Monte Carlo model for estimating team strength parameters
and computing win probabilities.

Model specification
-------------------
We use a Bayesian Bradley-Terry model:

    For each game (i vs j):
        P(i beats j) = σ(α_i − α_j + home_i * δ)

    where α_i ~ Normal(0, σ_α²) are latent team strengths,
    and δ ~ Normal(0, 1) is a home-court advantage parameter.

We sample the posterior P(α | games) via Metropolis-Hastings MCMC
(no external probabilistic programming library needed — pure NumPy/SciPy).

Then, for tournament simulation we:
  1. Draw K samples from the posterior
  2. For each sample, simulate the entire 64-team bracket
  3. Aggregate win counts → championship probabilities
"""

import numpy as np
import pandas as pd
from scipy.special import expit   # sigmoid σ(x)
from scipy.stats import norm
from typing import Optional


class MCMCModel:
    """
    Metropolis-Hastings sampler for Bradley-Terry team strength parameters.

    Parameters
    ----------
    teams_df  : DataFrame with columns [team, seed, strength, offensive_rating,
                defensive_rating, net_rating, efg_pct, to_pct, sos, win_pct ...]
    games_df  : DataFrame with columns [team1, team2, winner]
    n_warmup  : MCMC warmup / burn-in steps
    n_samples : MCMC sampling steps after warmup
    step_size : Metropolis-Hastings proposal standard deviation
    seed      : random seed
    """

    def __init__(
        self,
        teams_df:  pd.DataFrame,
        games_df:  pd.DataFrame,
        n_warmup:  int = 2000,
        n_samples: int = 4000,
        step_size: float = 0.08,
        seed: int = 0,
    ):
        self.teams_df  = teams_df.reset_index(drop=True)
        self.games_df  = games_df
        self.n_warmup  = n_warmup
        self.n_samples = n_samples
        self.step_size = step_size
        self.rng       = np.random.default_rng(seed)

        self.team_names: list[str] = list(teams_df["team"])
        self.n_teams    = len(self.team_names)
        self.idx        = {t: i for i, t in enumerate(self.team_names)}

        # Will be filled after fit()
        self.alpha_samples: Optional[np.ndarray] = None   # (n_samples, n_teams)
        self.acceptance_rate: float = 0.0
        self._prior_alpha: np.ndarray | None = None

    # ── prior construction ────────────────────────────────────────────────

    def _build_informative_prior(self) -> np.ndarray:
        """
        Build a data-driven prior mean for α from observable stats.
        Combines net rating, SOS, eFG%, turnover %, and win percentage.
        All features z-scored then combined with learned weights.
        """
        df = self.teams_df.copy()

        features = []
        weights  = []

        for col, w in [
            ("net_rating",  0.40),
            ("sos",         0.15),
            ("efg_pct",     0.15),
            ("win_pct",     0.15),
            ("kenpom_rank", -0.15),   # lower rank = better
        ]:
            if col in df.columns:
                vals = df[col].fillna(df[col].median()).values.astype(float)
                std  = vals.std() or 1.0
                features.append((vals - vals.mean()) / std)
                weights.append(w)

        if not features:
            return np.zeros(self.n_teams)

        features = np.column_stack(features)
        weights  = np.array(weights)
        weights  = weights / weights.sum()
        prior_mean = features @ weights          # shape (n_teams,)
        # Scale to reasonable latent-strength range
        prior_mean = prior_mean / (prior_mean.std() or 1.0)
        return prior_mean

    # ── log-likelihood & log-posterior ───────────────────────────────────

    def _log_likelihood(self, alpha: np.ndarray) -> float:
        """
        Sum of log P(outcome | alpha) over all observed games.
        Uses Bradley-Terry: P(i wins) = σ(alpha_i - alpha_j).
        """
        ll = 0.0
        for _, row in self.games_df.iterrows():
            t1, t2, winner = row.get("team1"), row.get("team2"), row.get("winner")
            if t1 not in self.idx or t2 not in self.idx:
                continue
            i, j = self.idx[t1], self.idx[t2]
            diff = alpha[i] - alpha[j]
            if winner == t1:
                ll += np.log(expit(diff) + 1e-10)
            else:
                ll += np.log(expit(-diff) + 1e-10)
        return ll

    def _log_prior(self, alpha: np.ndarray, prior_mean: np.ndarray) -> float:
        """Normal prior: α_i ~ N(prior_mean_i, 1)."""
        return float(norm.logpdf(alpha, loc=prior_mean, scale=1.0).sum())

    def _log_posterior(self, alpha: np.ndarray, prior_mean: np.ndarray) -> float:
        return self._log_likelihood(alpha) + self._log_prior(alpha, prior_mean)

    # ── Metropolis-Hastings sampler ───────────────────────────────────────

    def fit(self, verbose: bool = True) -> "MCMCModel":
        """
        Run Metropolis-Hastings to sample the posterior over team strengths.
        Uses component-wise (block) updates for efficiency.
        """
        prior_mean = self._build_informative_prior()
        self._prior_alpha = prior_mean

        # Initialise at prior mean
        alpha_curr = prior_mean.copy()
        lp_curr    = self._log_posterior(alpha_curr, prior_mean)

        total   = self.n_warmup + self.n_samples
        samples = np.zeros((self.n_samples, self.n_teams))
        accepts = 0

        if verbose:
            print(f"[MCMC] Starting Metropolis-Hastings  "
                  f"(warmup={self.n_warmup}, samples={self.n_samples})")

        for step in range(total):
            # Propose: perturb all α simultaneously
            proposal = alpha_curr + self.rng.normal(0, self.step_size, self.n_teams)
            # Identifiability: fix sum(α)=0
            proposal -= proposal.mean()

            lp_prop = self._log_posterior(proposal, prior_mean)
            log_accept = lp_prop - lp_curr

            if np.log(self.rng.random() + 1e-300) < log_accept:
                alpha_curr = proposal
                lp_curr    = lp_prop
                if step >= self.n_warmup:
                    accepts += 1

            if step >= self.n_warmup:
                samples[step - self.n_warmup] = alpha_curr.copy()

            if verbose and (step + 1) % 1000 == 0:
                pct = (step + 1) / total * 100
                phase = "warmup" if step < self.n_warmup else "sampling"
                print(f"  [{phase}] step {step+1:>5}/{total}  ({pct:4.0f}%)")

        self.alpha_samples   = samples
        self.acceptance_rate = accepts / self.n_samples
        if verbose:
            print(f"[MCMC] Done. Acceptance rate = {self.acceptance_rate:.2%}")
        return self

    # ── win-probability utilities ─────────────────────────────────────────

    def win_probability(self, team_a: str, team_b: str) -> float:
        """
        P(team_a beats team_b) averaged over posterior samples.
        """
        if self.alpha_samples is None:
            raise RuntimeError("Call .fit() first.")
        ia, ib = self.idx[team_a], self.idx[team_b]
        diffs = self.alpha_samples[:, ia] - self.alpha_samples[:, ib]
        return float(expit(diffs).mean())

    def posterior_summary(self) -> pd.DataFrame:
        """
        DataFrame of posterior mean / std / 5th–95th pct for each team's α.
        """
        if self.alpha_samples is None:
            raise RuntimeError("Call .fit() first.")
        rows = []
        for i, team in enumerate(self.team_names):
            s = self.alpha_samples[:, i]
            rows.append({
                "team":  team,
                "alpha_mean": s.mean(),
                "alpha_std":  s.std(),
                "alpha_p5":   np.percentile(s, 5),
                "alpha_p95":  np.percentile(s, 95),
            })
        df = pd.DataFrame(rows).sort_values("alpha_mean", ascending=False).reset_index(drop=True)
        df["strength_rank"] = range(1, len(df) + 1)
        return df
