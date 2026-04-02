"""
CLI for March Madness Predictor using Typer.
"""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

from marchmadness import MarchMadnessPredictor, Config

app = typer.Typer(help="🏀 March Madness MCMC Predictor CLI")
console = Console()

predictor_app = typer.Typer(help="Run prediction pipeline")
matchup_app = typer.Typer(help="Query matchup probabilities")
app.add_typer(predictor_app, name="predict")
app.add_typer(matchup_app, name="matchup")


@predictor_app.command("run")
def run_prediction(
    season: int = typer.Option(2025, "--season", "-s", help="Season end year"),
    n_sims: int = typer.Option(
        10000, "--n-sims", "-n", help="Number of tournament simulations"
    ),
    n_warmup: int = typer.Option(2000, "--n-warmup", help="MCMC warmup iterations"),
    n_samples: int = typer.Option(4000, "--n-samples", help="MCMC sampling iterations"),
    step_size: float = typer.Option(0.08, "--step-size", help="MCMC step size"),
    seed: int = typer.Option(0, "--seed", help="Random seed"),
    offline: bool = typer.Option(
        True, "--offline/--live", help="Use offline (synthetic) data"
    ),
    output: str = typer.Option(
        "march_madness.png", "--output", "-o", help="Output file path"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", help="Print progress messages"
    ),
):
    """Run the full March Madness prediction pipeline."""
    cfg = Config.load(config) if config else None

    console.print(f"[bold cyan]🏀 March Madness MCMC Predictor[/bold cyan]")
    console.print(f"Season: {season} | Simulations: {n_sims:,}")

    predictor = MarchMadnessPredictor(season=season, verbose=verbose, config=cfg)

    console.print("\n[1/4] Loading data...")
    predictor.load_data(offline=offline)

    console.print("[2/4] Fitting MCMC model...")
    predictor.fit_mcmc(
        n_warmup=n_warmup, n_samples=n_samples, step_size=step_size, seed=seed
    )

    console.print("[3/4] Running tournament simulations...")
    results = predictor.run_tournament_simulation(n_simulations=n_sims, seed=seed)

    console.print("[4/4] Generating report...")
    predictor.report(sim_results=results, save_path=output)

    console.print(f"\n[bold green]✅ Results saved to: {output}[/bold green]")


@predictor_app.command("summary")
def show_summary(
    season: int = typer.Option(2025, "--season", "-s"),
    n_sims: int = typer.Option(1000, "--n-sims", "-n"),
    offline: bool = typer.Option(True, "--offline/--live"),
    top_n: int = typer.Option(10, "--top", "-t", help="Number of teams to show"),
):
    """Run prediction and show summary table without generating plot."""
    predictor = MarchMadnessPredictor(season=season, verbose=False)
    predictor.load_data(offline=offline)
    predictor.fit_mcmc(n_warmup=500, n_samples=1000)
    results = predictor.run_tournament_simulation(n_simulations=n_sims)

    table = Table(title=f"March Madness {season} - Championship Probabilities")
    table.add_column("Rank", style="cyan")
    table.add_column("Team", style="green")
    table.add_column("Seed", style="yellow")
    table.add_column("Champion %", justify="right")
    table.add_column("Final 4 %", justify="right")
    table.add_column("Elite 8 %", justify="right")

    seed_map = dict(zip(predictor.teams_df["team"], predictor.teams_df["seed"]))

    champ_probs = results["championship_probs"]
    ff_probs = results["final_four_probs"]
    e8_probs = results["elite_eight_probs"]

    for rank, team in enumerate(champ_probs.head(top_n).index, 1):
        table.add_row(
            str(rank),
            team,
            str(seed_map.get(team, "?")),
            f"{champ_probs.get(team, 0) * 100:.1f}%",
            f"{ff_probs.get(team, 0) * 100:.1f}%",
            f"{e8_probs.get(team, 0) * 100:.1f}%",
        )

    console.print(table)


@matchup_app.command("prob")
def matchup_prob(
    team_a: str = typer.Argument(..., help="First team name"),
    team_b: str = typer.Argument(..., help="Second team name"),
    season: int = typer.Option(2025, "--season", "-s"),
    offline: bool = typer.Option(True, "--offline/--live"),
):
    """Get win probability for a matchup."""
    predictor = MarchMadnessPredictor(season=season, verbose=False)
    predictor.load_data(offline=offline)
    predictor.fit_mcmc(n_warmup=500, n_samples=1000)

    try:
        prob = predictor.win_probability(team_a, team_b)
        console.print(f"[bold]{team_a}[/bold] vs [bold]{team_b}[/bold]")
        console.print(f"P({team_a} wins) = [green]{prob:.1%}[/green]")
        console.print(f"P({team_b} wins) = [red]{1 - prob:.1%}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("config")
def show_config(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
):
    """Show current configuration."""
    cfg = Config.load(config) if config else Config()

    console.print("[bold]Data Configuration[/bold]")
    console.print(f"  Cache dir: {cfg.data.cache_dir}")
    console.print(f"  Offline fallback: {cfg.data.offline_fallback}")
    console.print(f"  Timeout: {cfg.data.timeout_seconds}s")

    console.print("\n[bold]Model Configuration[/bold]")
    console.print(f"  MCMC warmup: {cfg.model.mcmc_warmup}")
    console.print(f"  MCMC samples: {cfg.model.mcmc_samples}")
    console.print(f"  Step size: {cfg.model.mcmc_step_size}")
    console.print(f"  N simulations: {cfg.model.n_simulations}")

    console.print("\n[bold]Visualizer Configuration[/bold]")
    console.print(f"  DPI: {cfg.visualizer.dpi}")
    console.print(f"  Figure size: {cfg.visualizer.figure_size}")
    console.print(f"  Theme: {cfg.visualizer.theme}")


if __name__ == "__main__":
    app()
