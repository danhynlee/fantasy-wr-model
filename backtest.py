from model import load_season, build_wr_table, score_wrs, get_ppr_outcomes, WEIGHTS
import polars as pl

def run_backtest(season: int, weights: dict) -> dict:
  data = load_season(season)

  # Baseline: build WR table and score for weeks 1-4, then get actual PPR outcomes for weeks 5-17
  wrs_wk1_4 = build_wr_table(data, start_week=1, end_week=4)
  scores_wk1_4 = score_wrs(wrs_wk1_4, WEIGHTS)
  ppr_stats_wk1_4 = get_ppr_outcomes(data, start_week=1, end_week=4)


  # Actual: get actual PPR outcomes for weeks 5-17, filter to players with at least 6 games played
  actual_stats_wk5_17 = get_ppr_outcomes(data, start_week=5, end_week=17)
  actual_stats_wk5_17 = actual_stats_wk5_17.filter(pl.col("games_played") >= 6)


  # Join the model scores, baseline PPR, and actual PPR outcomes on player_id (inner join to only include players that exist in all three tables)
  model = scores_wk1_4.select("player_id", "player_name", "team", "score")
  baseline = ppr_stats_wk1_4.select("player_id", pl.col("ppr_pg").alias("ppr_pg_early"))
  actual = actual_stats_wk5_17.select("player_id", pl.col("ppr_pg").alias("ppr_pg_actual"), pl.col("games_played").alias("games_played_actual"))
  results = model.join(baseline, on="player_id", how="inner").join(actual, on="player_id", how="inner")

  return {
    "season": season,
    "players": results.height,
    "model": results.select(pl.corr("score", "ppr_pg_actual", method="spearman")).item(),
    "baseline": results.select(pl.corr("ppr_pg_early", "ppr_pg_actual", method="spearman")).item(),
  }

rows = [run_backtest(season, WEIGHTS) for season in [2021, 2022, 2023, 2024]]
print(pl.DataFrame(rows))