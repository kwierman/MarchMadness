#!/usr/bin/env python3
"""
run_prediction.py
-----------------
Run the full March Madness MCMC prediction pipeline.

Usage:
    python run_prediction.py                  # quick demo (offline, ~30 sec)
    python run_prediction.py --live           # try live data from internet
    python run_prediction.py --n-sims 20000  # more simulations (slower, more accurate)
"""

import sys
import os
import argparse

# Make sure the march_madness package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marchmadness import MarchMadnessPredictor


def main():
    parser = argparse.ArgumentParser(description="March Madness MCMC Predictor")
    parser.add_argument(
        "--season", type=int, default=2025, help="Season year (default: 2025)"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Number of tournament simulations (default: 10000)",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=2000, help="MCMC warmup steps (default: 2000)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=4000, help="MCMC sample steps (default: 4000)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Attempt to fetch live data (requires internet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="march_madness_2025.png",
        help="Output PNG file path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🏀  MARCH MADNESS MCMC PREDICTOR  🏀")
    print(f"  Season: {args.season}   Simulations: {args.n_sims:,}")
    print("=" * 60)

    predictor = MarchMadnessPredictor(season=args.season, verbose=True)

    # ── 1. Data ──────────────────────────────────────────────────────────
    predictor.load_data(offline=not args.live)

    # ── 2. MCMC ──────────────────────────────────────────────────────────
    predictor.fit_mcmc(
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
    )

    # Show posterior summary
    print("\n[Posterior Strength Rankings — Top 12]")
    ps = predictor.posterior_summary().head(12)
    print(
        ps[
            [
                "strength_rank",
                "team",
                "alpha_mean",
                "alpha_std",
                "alpha_p5",
                "alpha_p95",
            ]
        ].to_string(index=False)
    )

    # ── 3. Tournament simulation ──────────────────────────────────────────
    sim_results = predictor.run_tournament_simulation(n_simulations=args.n_sims)

    # ── 4. Report ─────────────────────────────────────────────────────────
    predictor.report(sim_results, save_path=args.output)

    # ── 5. Sample matchup queries ─────────────────────────────────────────
    print("\n[Head-to-Head Win Probabilities — Sample Matchups]")
    sample_matchups = [
        ("Auburn", "Duke"),
        ("Houston", "Florida"),
        ("Duke", "Michigan St"),
        ("Tennessee", "Kentucky"),
        ("Gonzaga", "UConn"),
    ]
    for ta, tb in sample_matchups:
        try:
            p = predictor.win_probability(ta, tb)
            print(f"  {ta:>18} vs {tb:<18} →  P({ta} wins) = {p:.1%}")
        except Exception as e:
            print(f"  [{ta} vs {tb}] skipped: {e}")

    print(f"\n✅  Report saved to: {args.output}")


if __name__ == "__main__":
    main()
