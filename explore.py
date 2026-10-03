import nflreadpy as nfl
import polars as pl

pl.Config.set_tbl_rows(20)

stats = nfl.load_player_stats([2026])

wrs = stats.filter(
  (pl.col("position") == "WR") & 
  (pl.col("season_type") == "REG")
)
print(wrs.shape)

wr_stats = wrs.group_by("player_id").agg(
  pl.col("player_display_name").first().alias("player_name"),
  pl.col("team").last().alias("team"),
  pl.col("game_id").count().alias("games_played"),
  pl.col("target_share").mean().alias("target_share"),
  pl.col("air_yards_share").mean().alias("air_yards_share"),
  pl.col("receiving_tds").sum().alias("receiving_tds"),
  pl.col("targets").sum().alias("targets"),
  pl.col("receptions").sum().alias("receptions"),
  pl.col("receiving_yards").sum().alias("receiving_yards"),
  pl.col("receiving_yards_after_catch").sum().alias("YAC"),
)

wr_stats = wr_stats.with_columns(
  (pl.col("YAC") / pl.col("receptions")).alias("yac_per_rec")
)

wr_stats_filtered = wr_stats.filter(pl.col("target_share") >= .1).filter(pl.col("games_played") >= 3)
wr_stats_sorted = wr_stats_filtered.sort("target_share", descending=True)
print(wr_stats_sorted.shape)
print(wr_stats_sorted.head(10))



pbp = nfl.load_pbp([2026])

redzone_plays = pbp.filter(
  (pl.col("yardline_100") <= 20) &
  (pl.col("two_point_attempt") == 0)
)

# exclude plays with penalties
wr_redzone_targets = (
  redzone_plays
    .filter((pl.col("play_type") == "pass") & (pl.col("receiver_id").is_not_null()))
    .group_by("receiver_id")
    .agg(pl.len().alias("redzone_targets"))
)

wr_stats_final = wr_stats_sorted.join(wr_redzone_targets, left_on="player_id", right_on="receiver_id", how="left")
wr_stats_final = wr_stats_final.with_columns(
  pl.col("redzone_targets").fill_null(0)
)

snaps = nfl.load_snap_counts([2026])
players = nfl.load_players()

players_id = players.select(
  pl.col("gsis_id"),
  pl.col("pfr_id"),
)

snaps_with_gsis = snaps.join(players_id, left_on="pfr_player_id", right_on="pfr_id", how="left")

snaps_with_gsis_filtered = (
  snaps_with_gsis
    .filter(pl.col("game_type") == "REG")
    .group_by("gsis_id")
    .agg(pl.col("offense_pct").mean().alias("avg_offense_pct"))
)

wr_stats_final_with_snaps = wr_stats_final.join(snaps_with_gsis_filtered, left_on="player_id", right_on="gsis_id", how="left")
print(wr_stats_final_with_snaps.shape)
print(wr_stats_final_with_snaps.head(10))

wr_stats_prct = wr_stats_final_with_snaps.with_columns(
  (pl.col("target_share").rank() / pl.len() * 100).alias("target_share_prct"),
  ((pl.col("receiving_tds") / pl.col("games_played")).rank() / pl.len() * 100).alias("receiving_tds_prct"),
  ((pl.col("redzone_targets") / pl.col("games_played")).rank() / pl.len() * 100).alias("redzone_targets_prct"),
  (pl.col("avg_offense_pct").rank() / pl.len() * 100).alias("avg_offense_pct_prct"),
  (pl.col("air_yards_share").rank() / pl.len() * 100).alias("air_yards_share_prct"),
  (pl.col("yac_per_rec").rank() / pl.len() * 100).alias("yac_per_rec_prct"),
)

wr_stats_scored = wr_stats_prct.with_columns(
  (
    pl.col("target_share_prct") * 0.4 +
    pl.col("redzone_targets_prct") * 0.2 +
    pl.col("avg_offense_pct_prct") * 0.15 +
    pl.col("receiving_tds_prct") * 0.10 +
    pl.col("air_yards_share_prct") * 0.1 +
    pl.col("yac_per_rec_prct") * 0.05
  ).alias("score")
)

# print(wr_stats_scored.sort("score", descending=True).select(
#   "player_name",
#   "team",
#   "score"
# ).head(15))

scored = wr_stats_scored.sort("score", descending=True).select(
    "player_name", "score",
    "target_share_prct", "redzone_targets_prct", "avg_offense_pct_prct",
    "air_yards_share_prct", "receiving_tds_prct", "yac_per_rec_prct",
)

print(scored.head(15))