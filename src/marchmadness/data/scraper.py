"""
data/scraper.py
---------------
Scrapes NCAA basketball data from public sources.
Falls back to realistic synthetic data when network is unavailable.

Real data sources:
  - sports-reference.com/cbb  (team stats, schedules)
  - ESPN API                   (live scores, rosters)
  - NCAA.com                   (official bracket, seedings)
"""

import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from marchmadness.config import Config

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

SPORTS_REF_BASE = "https://www.sports-reference.com/cbb"
ESPN_API_BASE = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
)


class DataScraper:
    """
    Fetches and caches team/player/game statistics for NCAA Tournament teams.

    Parameters
    ----------
    season : int
        Academic season end year (e.g. 2025 for 2024-25 season).
    cache_dir : str
        Directory to store cached JSON/CSV files.
    use_cache : bool
        If True, load from cache when available.
    config : Config, optional
        Configuration object. If provided, takes precedence over individual parameters.
    """

    def __init__(
        self,
        season: int = 2025,
        cache_dir: str = ".mm_cache",
        use_cache: bool = True,
        config: Optional["Config"] = None,
        timeout: int = 15,
    ):
        if config is not None:
            self.season = config.season
            self.cache_dir = config.data.cache_dir
            self.use_cache = config.data.use_cache
            self.timeout = config.data.timeout_seconds
            self._config = config
        else:
            self.season = season
            self.cache_dir = cache_dir
            self.use_cache = use_cache
            self.timeout = timeout
            self._config = None

        os.makedirs(self.cache_dir, exist_ok=True)

        self.teams_df: pd.DataFrame | None = None
        self.games_df: pd.DataFrame | None = None
        self.players_df: pd.DataFrame | None = None

        self._sports_ref_base = SPORTS_REF_BASE
        self._espn_api_base = ESPN_API_BASE

    def set_urls(self, sports_reference_url: str = None, espn_api_url: str = None):
        """Set custom URLs for data sources."""
        if sports_reference_url:
            self._sports_ref_base = sports_reference_url
        if espn_api_url:
            self._espn_api_base = espn_api_url

    # ── public API ────────────────────────────────────────────────────────

    def fetch_all(self, offline: bool = False) -> dict[str, pd.DataFrame]:
        """
        Fetch team stats, historical game results, and player stats.

        Returns dict with keys: 'teams', 'games', 'players'
        """
        print(f"[DataScraper] Fetching {self.season} season data...")

        if offline or not self._network_available():
            print("[DataScraper] Network unavailable — using realistic synthetic data.")
            return self._synthetic_data()

        try:
            self.teams_df = self._fetch_team_stats()
            self.games_df = self._fetch_game_results()
            self.players_df = self._fetch_player_stats()
        except Exception as exc:
            print(
                f"[DataScraper] Scrape error ({exc}) — falling back to synthetic data."
            )
            return self._synthetic_data()

        return {
            "teams": self.teams_df,
            "games": self.games_df,
            "players": self.players_df,
        }

    # ── real scrapers ─────────────────────────────────────────────────────

    def _fetch_team_stats(self) -> pd.DataFrame:
        """Scrape per-100-possession team stats from sports-reference.com."""
        url = f"{self._sports_ref_base}/seasons/men/{self.season}-school-stats.html"
        cache_path = os.path.join(self.cache_dir, f"teams_{self.season}.csv")
        if self.use_cache and os.path.exists(cache_path):
            print(f"[DataScraper] Loaded teams from cache: {cache_path}")
            return pd.read_csv(cache_path)

        print(f"[DataScraper] Scraping team stats from {url}")
        resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", {"id": "basic_school_stats"})
        df = pd.read_html(str(table))[0]
        df.columns = ["_".join(c).strip("_") for c in df.columns]
        df = df[
            df["School"].notna() & ~df["School"].str.contains("School|Conf", na=True)
        ]
        df = df.rename(columns={"School": "team"})
        df.to_csv(cache_path, index=False)
        time.sleep(1)
        return df

    def _fetch_game_results(self) -> pd.DataFrame:
        """Scrape game-by-game results to build pairwise strength estimates."""
        url = f"{self._sports_ref_base}/seasons/men/{self.season}-schedule.html"
        cache_path = os.path.join(self.cache_dir, f"games_{self.season}.csv")
        if self.use_cache and os.path.exists(cache_path):
            return pd.read_csv(cache_path)

        print(f"[DataScraper] Scraping game results from {url}")
        resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        dfs = pd.read_html(resp.text)
        df = max(dfs, key=len)
        df.to_csv(cache_path, index=False)
        time.sleep(1)
        return df

    def _fetch_player_stats(self) -> pd.DataFrame:
        """Scrape individual player stats (pts/ast/reb/per)."""
        url = f"{self._sports_ref_base}/seasons/men/{self.season}-players.html"
        cache_path = os.path.join(self.cache_dir, f"players_{self.season}.csv")
        if self.use_cache and os.path.exists(cache_path):
            return pd.read_csv(cache_path)

        print(f"[DataScraper] Scraping player stats from {url}")
        resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table", {"id": "players"})
        if table is None:
            raise ValueError("Player table not found")
        df = pd.read_html(str(table))[0]
        df.to_csv(cache_path, index=False)
        time.sleep(1)
        return df

    # ── helpers ───────────────────────────────────────────────────────────

    def _network_available(self) -> bool:
        try:
            requests.get("https://www.google.com", timeout=3)
            return True
        except Exception:
            return False

    def _synthetic_data(self) -> dict[str, pd.DataFrame]:
        """
        Generate statistically realistic synthetic NCAA data for the 64 tournament teams.
        Stats are based on historical averages of actual tournament participants.
        """
        rng = np.random.default_rng(42)

        # 2025 realistic tournament bracket seeds & teams
        tournament_teams = [
            # (seed, team, conference, region)
            (1, "Auburn", "SEC", "East"),
            (1, "Duke", "ACC", "West"),
            (1, "Houston", "Big 12", "South"),
            (1, "Florida", "SEC", "Midwest"),
            (2, "Michigan St", "Big Ten", "East"),
            (2, "Alabama", "SEC", "West"),
            (2, "Tennessee", "SEC", "South"),
            (2, "St. John's", "Big East", "Midwest"),
            (3, "Iowa St", "Big 12", "East"),
            (3, "Wisconsin", "Big Ten", "West"),
            (3, "Kentucky", "SEC", "South"),
            (3, "Texas Tech", "Big 12", "Midwest"),
            (4, "Arizona", "Big 12", "East"),
            (4, "Maryland", "Big Ten", "West"),
            (4, "Purdue", "Big Ten", "South"),
            (4, "Baylor", "Big 12", "Midwest"),
            (5, "Oregon", "Big Ten", "East"),
            (5, "Michigan", "Big Ten", "West"),
            (5, "Clemson", "ACC", "South"),
            (5, "Memphis", "AAC", "Midwest"),
            (6, "BYU", "Big 12", "East"),
            (6, "Illinois", "Big Ten", "West"),
            (6, "Ole Miss", "SEC", "South"),
            (6, "Kansas", "Big 12", "Midwest"),
            (7, "Gonzaga", "WCC", "East"),
            (7, "UCLA", "Big Ten", "West"),
            (7, "Missouri", "SEC", "South"),
            (7, "UConn", "Big East", "Midwest"),
            (8, "Mississippi St", "SEC", "East"),
            (8, "Creighton", "Big East", "West"),
            (8, "Georgia", "SEC", "South"),
            (8, "Louisville", "ACC", "Midwest"),
            (9, "Oklahoma", "SEC", "East"),
            (9, "Texas", "SEC", "West"),
            (9, "Xavier", "Big East", "South"),
            (9, "North Carolina", "ACC", "Midwest"),
            (10, "New Mexico", "MWC", "East"),
            (10, "Vanderbilt", "SEC", "West"),
            (10, "Arkansas", "SEC", "South"),
            (10, "Wake Forest", "ACC", "Midwest"),
            (11, "VCU", "A-10", "East"),
            (11, "Drake", "MVC", "West"),
            (11, "St. Mary's", "WCC", "South"),
            (11, "San Diego St", "MWC", "Midwest"),
            (12, "UC San Diego", "Big West", "East"),
            (12, "High Point", "BSouth", "West"),
            (12, "Colorado St", "MWC", "South"),
            (12, "McNeese", "Slnd", "Midwest"),
            (13, "Yale", "Ivy", "East"),
            (13, "Bryant", "A-East", "West"),
            (13, "Liberty", "CUSA", "South"),
            (13, "Akron", "MAC", "Midwest"),
            (14, "Lipscomb", "ASUN", "East"),
            (14, "Eastern Wash", "BSky", "West"),
            (14, "Colgate", "Patriot", "South"),
            (14, "Montana", "BSky", "Midwest"),
            (15, "Robert Morris", "Horizon", "East"),
            (15, "Long Beach St", "Big West", "West"),
            (15, "Wofford", "SoCon", "South"),
            (15, "South Dakota St", "Summit", "Midwest"),
            (16, "Alabama St", "SWAC", "East"),
            (16, "Norfolk St", "MEAC", "West"),
            (16, "SIUE", "OVC", "South"),
            (16, "Central Conn St", "NEC", "Midwest"),
        ]

        n = len(tournament_teams)
        teams = []
        for seed, team, conf, region in tournament_teams:
            # Strength is heavily influenced by seed; add noise for realism
            base_strength = max(0.1, 1.0 - (seed - 1) * 0.055 + rng.normal(0, 0.04))

            # Realistic stat distributions calibrated to actual D1 tournament teams
            off_rtg = rng.normal(108 + (16 - seed) * 0.9, 2.5)  # offensive rating
            def_rtg = rng.normal(
                98 - (16 - seed) * 0.8, 2.5
            )  # defensive rating (lower=better)
            pace = rng.normal(69, 2.0)
            eff_fg = rng.normal(0.522 + (16 - seed) * 0.003, 0.015)
            to_pct = rng.normal(17.0 - (16 - seed) * 0.2, 1.0)
            orb_pct = rng.normal(31.0 + (16 - seed) * 0.2, 2.0)
            ft_rate = rng.normal(0.33, 0.03)
            sos = rng.normal(8.0 + (16 - seed) * 0.3, 1.5)  # strength of schedule
            ppg = rng.normal(75 + (16 - seed) * 0.4, 3.5)
            apg = rng.normal(14.5, 1.5)
            rpg = rng.normal(36.5, 2.0)
            three_pt = rng.normal(0.355 + (16 - seed) * 0.001, 0.02)
            wins = int(rng.normal(25 + (16 - seed) * 0.7, 2))
            losses = max(1, int(rng.normal(8 - (16 - seed) * 0.2, 2)))
            kenpom = rng.normal(15 + (16 - seed) * 2.0, 3.0)

            teams.append(
                {
                    "team": team,
                    "seed": seed,
                    "conference": conf,
                    "region": region,
                    "strength": round(base_strength, 4),
                    "offensive_rating": round(off_rtg, 1),
                    "defensive_rating": round(def_rtg, 1),
                    "net_rating": round(off_rtg - def_rtg, 1),
                    "pace": round(pace, 1),
                    "efg_pct": round(eff_fg, 3),
                    "to_pct": round(to_pct, 1),
                    "orb_pct": round(orb_pct, 1),
                    "ft_rate": round(ft_rate, 3),
                    "sos": round(sos, 2),
                    "ppg": round(ppg, 1),
                    "apg": round(apg, 1),
                    "rpg": round(rpg, 1),
                    "three_pt_pct": round(three_pt, 3),
                    "wins": wins,
                    "losses": losses,
                    "win_pct": round(wins / (wins + losses), 3),
                    "kenpom_rank": int(kenpom),
                }
            )

        teams_df = pd.DataFrame(teams)

        # ── Synthetic historical games ─────────────────────────────────
        games = []
        team_list = teams_df["team"].tolist()
        strengths = dict(zip(teams_df["team"], teams_df["strength"]))
        for _ in range(1200):
            t1, t2 = rng.choice(team_list, 2, replace=False)
            s1, s2 = strengths[t1], strengths[t2]
            prob_t1 = s1 / (s1 + s2)
            winner = t1 if rng.random() < prob_t1 else t2
            loser = t2 if winner == t1 else t1
            margin = int(abs(rng.normal((s1 - s2) * 25, 7)))
            games.append(
                {
                    "team1": t1,
                    "team2": t2,
                    "winner": winner,
                    "loser": loser,
                    "margin": margin,
                }
            )
        games_df = pd.DataFrame(games)

        # ── Synthetic player stats ─────────────────────────────────────
        players = []
        for _, row in teams_df.iterrows():
            n_players = 10
            for p in range(n_players):
                role_weight = max(0.1, 1.0 - p * 0.1)
                players.append(
                    {
                        "player": f"{row['team']}_P{p + 1}",
                        "team": row["team"],
                        "ppg": round(
                            rng.normal(row["ppg"] / 5 * role_weight * 1.5, 2), 1
                        ),
                        "rpg": round(rng.normal(row["rpg"] / 5 * role_weight, 1), 1),
                        "apg": round(rng.normal(row["apg"] / 5 * role_weight, 0.8), 1),
                        "fg_pct": round(
                            rng.normal(0.46 + row["seed"] * -0.003, 0.04), 3
                        ),
                        "three_pt_pct": round(rng.normal(0.35, 0.05), 3),
                        "per": round(
                            rng.normal(18 - row["seed"] * 0.3 + role_weight * 5, 3), 1
                        ),
                        "minutes": round(rng.normal(32 * role_weight, 4), 1),
                    }
                )
        players_df = pd.DataFrame(players)

        self.teams_df = teams_df
        self.games_df = games_df
        self.players_df = players_df

        return {"teams": teams_df, "games": games_df, "players": players_df}
