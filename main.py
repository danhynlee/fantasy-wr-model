from model import load_season, build_wr_table, score_wrs, WEIGHTS
import polars as pl

pl.Config.set_tbl_rows(20)

data = load_season(2026)
wrs = build_wr_table(data, start_week=1, end_week=4)
scored = score_wrs(wrs, WEIGHTS)

print(scored.sort("score", descending=True).select("player_name", "team", "score").head(15))