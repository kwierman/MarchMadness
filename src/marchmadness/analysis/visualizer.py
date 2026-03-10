"""
analysis/visualizer.py
-----------------------
Publication-quality visualisations for March Madness MCMC results.

Produces:
  1. Championship probability bar chart (top 16 teams)
  2. MCMC posterior density plot (top 8 team α distributions)
  3. Round-by-round survival matrix heat-map
  4. Seeding upset probability chart
  5. Head-to-head win probability matrix (top 8 teams)
  6. MCMC trace & autocorrelation diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from typing import Optional


# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = [
    "#1B4F72","#2980B9","#1ABC9C","#F39C12","#E74C3C",
    "#8E44AD","#27AE60","#D35400","#2C3E50","#16A085",
    "#C0392B","#2471A3","#117A65","#B7950B","#6C3483",
    "#1A5276",
]
BRACKET_ORANGE = "#E87722"
BRACKET_BLUE   = "#003087"


class TournamentVisualizer:
    """
    Wraps all plotting functionality for March Madness predictions.

    Parameters
    ----------
    teams_df       : full teams dataframe
    sim_results    : dict returned by BracketSimulator.run_simulations()
    posterior_df   : dataframe from MCMCModel.posterior_summary()
    alpha_samples  : np.ndarray (n_samples, n_teams)
    team_names     : list aligned with alpha_samples
    """

    def __init__(
        self,
        teams_df:      pd.DataFrame,
        sim_results:   dict,
        posterior_df:  pd.DataFrame,
        alpha_samples: np.ndarray,
        team_names:    list[str],
    ):
        self.teams_df      = teams_df
        self.sim_results   = sim_results
        self.posterior_df  = posterior_df
        self.alpha_samples = alpha_samples
        self.team_names    = team_names
        self.idx           = {t: i for i, t in enumerate(team_names)}

        # Seed lookup
        self.seed_map = dict(zip(teams_df["team"], teams_df["seed"]))

        plt.rcParams.update({
            "font.family":      "DejaVu Sans",
            "axes.spines.top":  False,
            "axes.spines.right":False,
            "axes.grid":        True,
            "grid.alpha":       0.3,
            "figure.dpi":       150,
        })

    # ── 1. Championship probability bar chart ─────────────────────────────

    def plot_championship_probs(self, ax: Optional[plt.Axes] = None,
                                 top_n: int = 16) -> plt.Axes:
        probs = self.sim_results["championship_probs"].head(top_n)
        teams = probs.index.tolist()
        vals  = probs.values * 100

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        seeds  = [self.seed_map.get(t, "?") for t in teams]
        labels = [f"({s}) {t}" for s, t in zip(seeds, teams)]
        colours = [BRACKET_ORANGE if i == 0 else BRACKET_BLUE if i < 4 else "#4A90D9"
                   for i in range(len(teams))]

        bars = ax.barh(labels[::-1], vals[::-1], color=colours[::-1], edgecolor="white", height=0.7)

        for bar, v in zip(bars, vals[::-1]):
            ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=8, color="#2C3E50")

        ax.set_xlabel("Championship Probability (%)", fontsize=11)
        ax.set_title("🏆  NCAA Championship Win Probability\n(MCMC Posterior Simulation)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim(0, max(vals) * 1.18)
        return ax

    # ── 2. Posterior density traces ───────────────────────────────────────

    def plot_posterior_densities(self, ax: Optional[plt.Axes] = None,
                                  top_n: int = 8) -> plt.Axes:
        top_teams = self.posterior_df.head(top_n)["team"].tolist()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        for i, team in enumerate(top_teams):
            if team not in self.idx:
                continue
            samples = self.alpha_samples[:, self.idx[team]]
            kde = gaussian_kde(samples, bw_method=0.3)
            xs  = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 300)
            ys  = kde(xs)
            colour = PALETTE[i % len(PALETTE)]
            seed   = self.seed_map.get(team, "?")
            ax.plot(xs, ys, color=colour, lw=2, label=f"({seed}) {team}")
            ax.fill_between(xs, ys, alpha=0.12, color=colour)

        ax.set_xlabel("Team Strength (α)", fontsize=11)
        ax.set_ylabel("Posterior Density", fontsize=11)
        ax.set_title("MCMC Posterior Distributions — Top Teams\n"
                     "Wider = more uncertain; rightward = stronger",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        return ax

    # ── 3. Round-by-round survival heat-map ──────────────────────────────

    def plot_survival_matrix(self, ax: Optional[plt.Axes] = None,
                              top_n: int = 20) -> plt.Axes:
        rounds = [
            ("R64→R32",   "round_32_probs"),
            ("Sweet 16",  "sweet_16_probs"),
            ("Elite 8",   "elite_eight_probs"),
            ("Final Four","final_four_probs"),
            ("Champion",  "championship_probs"),
        ]
        champ_order = self.sim_results["championship_probs"].head(top_n).index.tolist()

        matrix = np.zeros((len(champ_order), len(rounds)))
        for j, (_, key) in enumerate(rounds):
            prob_series = self.sim_results[key]
            for i, team in enumerate(champ_order):
                matrix[i, j] = prob_series.get(team, 0.0) * 100

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        cmap = LinearSegmentedColormap.from_list(
            "mm", ["#FFFFFF", "#2980B9", "#1B4F72"], N=256
        )
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100)

        seeds  = [self.seed_map.get(t, "?") for t in champ_order]
        labels = [f"({s}) {t}" for s, t in zip(seeds, champ_order)]
        ax.set_yticks(range(len(champ_order)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([r[0] for r in rounds], fontsize=9)

        for i in range(len(champ_order)):
            for j in range(len(rounds)):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if v > 50 else "#2C3E50")

        plt.colorbar(im, ax=ax, label="Probability (%)", shrink=0.8)
        ax.set_title("Round-by-Round Survival Probabilities\n(Top 20 Teams by Championship Odds)",
                     fontsize=12, fontweight="bold")
        return ax

    # ── 4. Upset probability by seed matchup ──────────────────────────────

    def plot_upset_probs(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        For each first-round seed matchup (1v16, 2v15 … 8v9),
        show the lower seed's upset probability.
        """
        matchups = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]

        upset_probs = []
        labels      = []
        for s_fav, s_dog in matchups:
            region_rows = self.teams_df
            favs = region_rows[region_rows["seed"] == s_fav]["team"].tolist()
            dogs = region_rows[region_rows["seed"] == s_dog]["team"].tolist()

            if not favs or not dogs:
                continue

            # Average across all regions
            probs = []
            for f, d in zip(favs, dogs):
                if f in self.idx and d in self.idx:
                    diffs = (self.alpha_samples[:, self.idx[d]]
                             - self.alpha_samples[:, self.idx[f]])
                    from scipy.special import expit
                    probs.append(float(expit(diffs).mean()))
            if probs:
                upset_probs.append(np.mean(probs) * 100)
                labels.append(f"{s_dog} upsets {s_fav}")

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        colours = ["#E74C3C" if p > 30 else "#F39C12" if p > 15 else "#27AE60"
                   for p in upset_probs]
        ax.bar(labels, upset_probs, color=colours, edgecolor="white")
        ax.axhline(50, color="grey", lw=0.8, linestyle="--", alpha=0.5, label="50% line")
        ax.set_ylabel("Upset Probability (%)")
        ax.set_title("First-Round Upset Probabilities by Seed Matchup",
                     fontsize=12, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)

        for bar, v in zip(ax.patches, upset_probs):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"{v:.1f}%", ha="center", fontsize=8)
        return ax

    # ── 5. Head-to-head win prob matrix ───────────────────────────────────

    def plot_h2h_matrix(self, ax: Optional[plt.Axes] = None, top_n: int = 8) -> plt.Axes:
        from scipy.special import expit
        top_teams = self.posterior_df.head(top_n)["team"].tolist()

        matrix = np.zeros((top_n, top_n))
        for i, ta in enumerate(top_teams):
            for j, tb in enumerate(top_teams):
                if i == j:
                    matrix[i, j] = 0.5
                elif ta in self.idx and tb in self.idx:
                    diffs = (self.alpha_samples[:, self.idx[ta]]
                             - self.alpha_samples[:, self.idx[tb]])
                    matrix[i, j] = float(expit(diffs).mean())

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 7))

        cmap = LinearSegmentedColormap.from_list(
            "h2h", ["#2980B9","#FFFFFF","#E74C3C"], N=256
        )
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)

        seeds  = [self.seed_map.get(t, "?") for t in top_teams]
        labels = [f"({s}) {t}" for s, t in zip(seeds, top_teams)]
        ax.set_xticks(range(top_n));  ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(top_n));  ax.set_yticklabels(labels, fontsize=8)

        for i in range(top_n):
            for j in range(top_n):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(v - 0.5) > 0.25 else "#2C3E50")

        plt.colorbar(im, ax=ax, label="P(row beats column)", shrink=0.85)
        ax.set_title("Head-to-Head Win Probability Matrix\n(Top 8 Teams by MCMC Strength)",
                     fontsize=12, fontweight="bold")
        return ax

    # ── 6. MCMC diagnostic traces ─────────────────────────────────────────

    def plot_mcmc_diagnostics(self, ax_trace: Optional[plt.Axes] = None,
                               ax_acf: Optional[plt.Axes] = None,
                               n_teams: int = 4) -> tuple:
        top_teams = self.posterior_df.head(n_teams)["team"].tolist()

        if ax_trace is None or ax_acf is None:
            fig, (ax_trace, ax_acf) = plt.subplots(1, 2, figsize=(14, 5))

        # Trace plot
        for i, team in enumerate(top_teams):
            if team not in self.idx:
                continue
            s = self.alpha_samples[:, self.idx[team]]
            ax_trace.plot(s, alpha=0.7, lw=0.8, color=PALETTE[i],
                          label=f"({self.seed_map.get(team,'?')}) {team}")
        ax_trace.set_xlabel("Sample")
        ax_trace.set_ylabel("α (team strength)")
        ax_trace.set_title("MCMC Trace Plot\n(should look like 'hairy caterpillar')",
                           fontsize=11, fontweight="bold")
        ax_trace.legend(fontsize=8)

        # Autocorrelation
        max_lag = 50
        for i, team in enumerate(top_teams):
            if team not in self.idx:
                continue
            s    = self.alpha_samples[:, self.idx[team]]
            s_z  = (s - s.mean()) / (s.std() or 1)
            acf  = np.correlate(s_z, s_z, mode="full")
            acf  = acf[len(acf)//2:][:max_lag] / acf[len(acf)//2]
            ax_acf.plot(acf, color=PALETTE[i], lw=1.5,
                        label=f"({self.seed_map.get(team,'?')}) {team}")
        ax_acf.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax_acf.fill_between(range(max_lag), -1.96/np.sqrt(len(s)), 1.96/np.sqrt(len(s)),
                             alpha=0.15, color="grey", label="95% CI")
        ax_acf.set_xlabel("Lag")
        ax_acf.set_ylabel("Autocorrelation")
        ax_acf.set_title("Sample Autocorrelation\n(should drop quickly to 0)",
                         fontsize=11, fontweight="bold")
        ax_acf.legend(fontsize=8)
        return ax_trace, ax_acf

    # ── master plot ───────────────────────────────────────────────────────

    def plot_all(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Produce a 6-panel summary figure.
        """
        fig = plt.figure(figsize=(22, 26))
        fig.patch.set_facecolor("#F8F9FA")

        gs = gridspec.GridSpec(4, 2, figure=fig,
                               hspace=0.45, wspace=0.35,
                               top=0.95, bottom=0.04, left=0.08, right=0.97)

        ax1 = fig.add_subplot(gs[0, :])    # full-width: championship probs
        ax2 = fig.add_subplot(gs[1, 0])    # posterior densities
        ax3 = fig.add_subplot(gs[1, 1])    # upset probs
        ax4 = fig.add_subplot(gs[2, :])    # survival matrix
        ax5 = fig.add_subplot(gs[3, 0])    # h2h matrix
        ax6_l = fig.add_subplot(gs[3, 1])  # trace (left half of right column)

        self.plot_championship_probs(ax=ax1)
        self.plot_posterior_densities(ax=ax2)
        self.plot_upset_probs(ax=ax3)
        self.plot_survival_matrix(ax=ax4)
        self.plot_h2h_matrix(ax=ax5)

        # Squeeze MCMC diagnostics into ax6 area
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[3, 1], wspace=0.45
        )
        ax6a = fig.add_subplot(inner_gs[0, 0])
        ax6b = fig.add_subplot(inner_gs[0, 1])
        fig.delaxes(ax6_l)
        self.plot_mcmc_diagnostics(ax_trace=ax6a, ax_acf=ax6b)

        n_sim = self.sim_results.get("n_simulations", "?")
        champ = self.sim_results["championship_probs"].idxmax()
        prob  = self.sim_results["championship_probs"].max() * 100
        fig.suptitle(
            f"🏀  2025 NCAA March Madness — MCMC Prediction Report\n"
            f"{n_sim:,} Tournament Simulations  •  "
            f"Predicted Champion: {champ}  ({prob:.1f}%)",
            fontsize=14, fontweight="bold", y=0.975,
            color=BRACKET_BLUE,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[Visualizer] Saved → {save_path}")

        return fig
