import nflreadpy as nfl
import polars as pl

WEIGHTS = {
    "target_share": 0.40,
    "rz_targets_pg": 0.20,
    "snap_share": 0.15,
    "air_yards_share": 0.10,
    "receiving_tds_pg": 0.10,
    "yac_per_rec": 0.05,
}


def load_season(season: int) -> dict:
    """Load all raw data for one season. Slow, so call it once and reuse."""
    return {
        "stats": nfl.load_player_stats([season]),
        "pbp": nfl.load_pbp([season]),
        "snaps": nfl.load_snap_counts([season]),
        "players": nfl.load_players(),
    }


def build_wr_table(data: dict, start_week: int, end_week: int, min_games: int = 2, min_target_share: float = 0.10) -> pl.DataFrame:
    """One row per fantasy-relevant WR, with all six stats over the given weeks."""
    # your existing code goes here

    # initialize the dataframes with function params
    stats = data["stats"].filter((pl.col("season_type") == "REG") & (pl.col("week").is_between(start_week, end_week)))
    pbp = data["pbp"].filter((pl.col("season_type") == "REG") & (pl.col("week").is_between(start_week, end_week)) & (pl.col("yardline_100") <= 20) & (pl.col("two_point_attempt") == 0))
    snaps = data["snaps"].filter((pl.col("game_type") == "REG") & (pl.col("week").is_between(start_week, end_week)))
    players = data["players"]

    wr_table = (stats
      .filter((pl.col("position") == "WR"))
      .group_by("player_id")
      .agg(
        pl.col("player_display_name").first().alias("player_name"),
        pl.col("team").last().alias("team"),
        pl.col("game_id").count().alias("games_played"),
        pl.col("target_share").mean().alias("target_share"),
        pl.col("air_yards_share").mean().alias("air_yards_share"),
        pl.col("receiving_tds").sum().alias("receiving_tds"),
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("receiving_yards").sum().alias("receiving_yards"),
        pl.col("receiving_yards_after_catch").sum().alias("yards_after_catch"),
      )
    )

    wr_table = wr_table.with_columns(
      (pl.col("yards_after_catch") / pl.col("receptions")).alias("yac_per_rec")
    )

    # filter for WRs with at least min_games and min_target_share
    wr_table = wr_table.filter((pl.col("games_played") >= min_games) & (pl.col("target_share") >= min_target_share))

    # get all redzone targets for each WR
    rz_targets = (pbp
      .filter((pl.col("play_type") == "pass") & (pl.col("receiver_id").is_not_null()))
      .group_by("receiver_id")
      .agg(pl.len().alias("rz_targets"))
    )

    # get the gsis_id and pfr_id for each player
    players_id = players.select(
      pl.col("gsis_id"),
      pl.col("pfr_id"),
    )

    # join redzone targets to the WR table and fill nulls with 0
    wr_stats = wr_table.join(rz_targets, left_on="player_id", right_on="receiver_id", how="left")
    wr_stats = wr_stats.with_columns(
      pl.col("rz_targets").fill_null(0)
    )

    # join snap counts to the WR table and calculate snap share
    snaps_ids = snaps.join(players_id, left_on="pfr_player_id", right_on="pfr_id", how="left")

    snaps_ids = (
      snaps_ids
        .group_by("gsis_id")
        .agg(pl.col("offense_pct").mean().alias("snap_share"))
    )

    wr_table_final = wr_stats.join(snaps_ids, left_on="player_id", right_on="gsis_id", how="left")

    return wr_table_final


def score_wrs(wr_table: pl.DataFrame, weights: dict) -> pl.DataFrame:
    """Add percentile columns and a weighted score."""
    # your percentile + score code goes here

    wr_prct = wr_table.with_columns(
      (pl.col("target_share").rank() / pl.len() * 100).alias("target_share_prct"),
      ((pl.col("receiving_tds") / pl.col("games_played")).rank() / pl.len() * 100).alias("receiving_tds_pg_prct"),
      ((pl.col("rz_targets") / pl.col("games_played")).rank() / pl.len() * 100).alias("rz_targets_pg_prct"),
      (pl.col("snap_share").rank() / pl.len() * 100).alias("snap_share_prct"),
      (pl.col("air_yards_share").rank() / pl.len() * 100).alias("air_yards_share_prct"),
      (pl.col("yac_per_rec").rank() / pl.len() * 100).alias("yac_per_rec_prct"),
    )

    wr_scored = wr_prct.with_columns(
      (
        pl.col("target_share_prct") * weights["target_share"] +
        pl.col("rz_targets_pg_prct") * weights["rz_targets_pg"] +
        pl.col("snap_share_prct") * weights["snap_share"] +
        pl.col("receiving_tds_pg_prct") * weights["receiving_tds_pg"] +
        pl.col("air_yards_share_prct") * weights["air_yards_share"] +
        pl.col("yac_per_rec_prct") * weights["yac_per_rec"]
      ).alias("score")
    )

    return wr_scored


def get_ppr_outcomes(data: dict, start_week: int, end_week: int) -> pl.DataFrame:
    """Actual PPR points per game for each player over the given weeks."""

    stats = data["stats"].filter((pl.col("season_type") == "REG") & (pl.col("week").is_between(start_week, end_week)))

    ppr_stats = stats.filter((pl.col("position") == "WR")).group_by("player_id").agg(
        pl.col("player_display_name").first().alias("player_name"),
        pl.col("fantasy_points_ppr").mean().alias("ppr_pg"),
        pl.col("game_id").count().alias("games_played"),
    )

    return ppr_stats