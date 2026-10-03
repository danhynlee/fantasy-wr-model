from model import load_season, build_wr_table, score_wrs, get_ppr_outcomes, WEIGHTS
import polars as pl

pl.Config.set_tbl_rows(20)

data = load_season(2026)
wrs = build_wr_table(data, start_week=1, end_week=18, min_games=1, min_target_share=0.05)
scored = score_wrs(wrs, WEIGHTS)

ppr_stats = get_ppr_outcomes(data, start_week=1, end_week=4) 

print(scored.sort("score", descending=True).select("player_name", "team", "score").head(15))

print(ppr_stats.sort("ppr_pg", descending=True).select("player_name", "ppr_pg", "games_played").head(15))