import pandas as pd
import numpy as np

SEASON_LIMIT = 25  # Number of seasons to include
SEASON_START_YEAR = 2001  # Filter 2001 <= year <= 2024 for last 25 complete seasons (2000-01 to 2024-25)
SEASON_END_YEAR = 2024


def load_and_clean_data():
    """Load CSV files and filter out players with missing critical demographic data"""
    print("Loading data files...")
    players = pd.read_csv("data/Players.csv")
    player_stats = pd.read_csv("data/PlayerStatistics.csv")
    team_stats = pd.read_csv("data/TeamStatistics.csv")

    print(
        f"Loaded {len(players)} players, {len(player_stats)} player stats, {len(team_stats)} team stats"
    )

    # Extract year and month from gameDateTimeEst (format: YYYY-MM-DD HH:MM:SS+TZ)
    for df in [player_stats, team_stats]:
        df["year"] = df["gameDateTimeEst"].str[:4].astype(int)
        df["month"] = df["gameDateTimeEst"].str[5:7].astype(int)

    # Step 1: Filter to last 25 complete seasons (2000-01 to 2024-25)
    print(f"\nFiltering to last {SEASON_LIMIT} seasons ({SEASON_START_YEAR} <= year <= {SEASON_END_YEAR})")
    print(f"Season range: {SEASON_START_YEAR - 1}-01 to {SEASON_END_YEAR}-25")

    # Record original counts
    original_player_stats = len(player_stats)
    original_team_stats = len(team_stats)
    original_players = len(players)

    # Filter stats by year
    player_stats = player_stats[(player_stats['year'] >= SEASON_START_YEAR) & (player_stats['year'] <= SEASON_END_YEAR)]
    team_stats = team_stats[(team_stats['year'] >= SEASON_START_YEAR) & (team_stats['year'] <= SEASON_END_YEAR)]

    # Filter players to only those appearing in filtered player_stats
    valid_player_ids = player_stats['personId'].unique()
    players = players[players['personId'].isin(valid_player_ids)]

    # Log season filtering results
    stats_reduced_pct = (1 - len(player_stats) / original_player_stats) * 100
    team_reduced_pct = (1 - len(team_stats) / original_team_stats) * 100
    players_reduced_pct = (1 - len(players) / original_players) * 100

    print(f"Player stats: {original_player_stats:,} → {len(player_stats):,} ({stats_reduced_pct:.1f}% reduction)")
    print(f"Team stats:   {original_team_stats:,} → {len(team_stats):,} ({team_reduced_pct:.1f}% reduction)")
    print(f"Players:      {original_players:,} → {len(players):,} ({players_reduced_pct:.1f}% reduction)")

    # Step 2: Filter out players with missing critical fields
    players_before_missing = len(players)
    players = players.dropna(subset=["country", "height", "bodyWeight"])
    filtered_missing = players_before_missing - len(players)
    print(f"Filtered {filtered_missing} players with missing demographic data")

    return players, player_stats, team_stats


def get_nba_season(year, month):
    """
    Convert calendar year to NBA season year.
    NBA seasons run Oct (Year) -> Jun (Year+1)

    Examples:
    - Nov 2024 -> 2024-25 season
    - Feb 2025 -> 2024-25 season
    - Oct 2025 -> 2025-26 season
    """
    if month >= 10:  # October onwards
        return f"{year}-{str(year + 1)[-2:]}"
    else:  # January - September
        return f"{year - 1}-{str(year)[-2:]}"


def aggregate_seasonal_stats(player_stats, players):
    """Aggregate player statistics by NBA season"""
    print("Aggregating seasonal statistics...")

    # Join player demographics
    merged = player_stats.merge(
        players[["personId", "country", "height", "bodyWeight"]],
        on="personId",
        how="inner",
    )
    print(f"Merged dataset: {len(merged)} records after joining")

    # Extract NBA season
    merged["season"] = merged.apply(
        lambda x: get_nba_season(x["year"], x["month"]), axis=1
    )

    # Historical aggregation per season
    seasonal_agg = (
        merged.groupby("season")
        .agg(
            {
                "threePointersAttempted": "mean",
                "threePointersMade": "mean",
                "threePointersPercentage": "mean",
                "points": "sum",
                "assists": "sum",
                "reboundsTotal": "sum",
                "steals": "sum",
                "blocks": "sum",
                "turnovers": "sum",
                "freeThrowsMade": "sum",
                "fieldGoalsMade": "sum",
            }
        )
        .reset_index()
    )

    # Convert percentages to decimal format (0-1)
    seasonal_agg["threePointersPercentage"] = seasonal_agg[
        "threePointersPercentage"
    ].fillna(0)

    # Sort by season
    seasonal_agg = seasonal_agg.sort_values("season").reset_index(drop=True)

    print(f"Created {len(seasonal_agg)} season aggregations")

    return merged, seasonal_agg


def aggregate_by_country(merged):
    """Aggregate player count by country and season"""
    print("Aggregating by country...")

    country_agg = (
        merged.groupby(["season", "country"])
        .agg({"personId": "nunique"})
        .rename(columns={"personId": "player_count"})
        .reset_index()
    )

    print(f"Created {len(country_agg)} country-season combinations")

    return country_agg


def aggregate_team_stats(team_stats):
    """Aggregate team statistics by season"""
    print("Aggregating team statistics...")

    # Extract NBA season
    team_stats["season"] = team_stats.apply(
        lambda x: get_nba_season(x["year"], x["month"]), axis=1
    )

    # Aggregate per team per season
    team_agg = (
        team_stats.groupby(["season", "teamName"])
        .agg(
            {
                "threePointersAttempted": "sum",
                "threePointersMade": "sum",
                "threePointersPercentage": "mean",
                "teamScore": "sum",
                "fieldGoalsPercentage": "mean",
                "freeThrowsPercentage": "mean",
            }
        )
        .reset_index()
    )

    # Handle NaN values
    team_agg["threePointersPercentage"] = team_agg["threePointersPercentage"].fillna(0)

    print(f"Created {len(team_agg)} team-season combinations")

    return team_agg


def get_top_scorers(merged, season):
    """Get top 10 scorers for a specific season"""
    season_data = merged[merged["season"] == season]

    if len(season_data) == 0:
        return pd.DataFrame()

    top_10 = (
        season_data.groupby(["firstName", "lastName", "personId"])
        .agg({"points": "sum"})
        .nlargest(10, "points")
        .reset_index()
    )

    top_10["rank"] = range(1, 11)
    top_10["player"] = top_10["firstName"] + " " + top_10["lastName"]

    return top_10[["rank", "player", "points", "personId"]]


def get_season_comparison(merged, season_a, season_b):
    """Get comparison data between two seasons for head-to-head analysis"""
    stats = ["points", "assists", "reboundsTotal", "steals", "blocks", "turnovers"]

    season_a_data = merged[merged["season"] == season_a]
    season_b_data = merged[merged["season"] == season_b]

    if len(season_a_data) == 0 or len(season_b_data) == 0:
        return None

    # Calculate means
    season_a_means = season_a_data[stats].mean()
    season_b_means = season_b_data[stats].mean()

    # Shooting percentages
    shooting = [
        "fieldGoalsPercentage",
        "threePointersPercentage",
        "freeThrowsPercentage",
    ]
    season_a_shooting = season_a_data[shooting].mean()
    season_b_shooting = season_b_data[shooting].mean()

    # 3-pointer evolution
    three_pa_season_a = season_a_data["threePointersAttempted"].sum()
    three_pa_season_b = season_b_data["threePointersAttempted"].sum()

    if three_pa_season_a > 0:
        three_diff_pct = (
            (three_pa_season_b - three_pa_season_a) / three_pa_season_a * 100
        )
    else:
        three_diff_pct = 0

    return {
        "radar": {
            season_a: season_a_means.values.tolist(),
            season_b: season_b_means.values.tolist(),
            "labels": stats,
        },
        "shooting": {
            season_a: (season_a_shooting * 100).tolist(),
            season_b: (season_b_shooting * 100).tolist(),
            "labels": shooting,
        },
        "three_pointer_diff": three_diff_pct,
    }


def validate_season_for_stats(season, stat_type="three-pointer"):
    """Check if season has valid data for specific stat type"""
    season_year = int(season.split("-")[0])

    if stat_type == "three-pointer" and season_year < 1979:
        return False, "3-Pointer statistics not available before 1979"

    return True, None


def get_available_seasons_for_stat(all_seasons, stat_type="three-pointer"):
    """Get seasons with valid data for specific stat"""
    if stat_type == "three-pointer":
        return [s for s in all_seasons if int(s.split("-")[0]) >= 1979]

    return all_seasons


def load_all_data():
    """
    Main function to load and process all data for the dashboard
    Returns a dictionary with all processed datasets
    """
    print("=" * 60)
    print("NBA Dashboard - Data Preprocessing")
    print("=" * 60)

    # Step 1: Load and clean
    players, player_stats, team_stats = load_and_clean_data()

    # Step 2: Aggregate seasonal stats
    merged, seasonal_agg = aggregate_seasonal_stats(player_stats, players)

    # Step 3: Aggregate by country
    country_agg = aggregate_by_country(merged)

    # Step 4: Aggregate team stats
    team_agg = aggregate_team_stats(team_stats)

    # Get available seasons
    available_seasons = sorted(seasonal_agg["season"].unique())

    print("=" * 60)
    print(f"Data preprocessing complete!")
    print(f"Available seasons: {available_seasons[0]} to {available_seasons[-1]}")
    print(f"Total seasons: {len(available_seasons)}")
    print("=" * 60)

    return {
        "merged": merged,
        "seasonal_agg": seasonal_agg,
        "country_agg": country_agg,
        "team_agg": team_agg,
        "available_seasons": available_seasons,
    }
