from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd, time, pathlib, os

OUTPUT = pathlib.Path("data/raw")
OUTPUT.mkdir(parents=True, exist_ok=True)

def get_season_game_ids(season="2022-23"):
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",   # NBA
    )
    games = finder.get_data_frames()[0]
    # deduplicate: each game appears twice (once per team)
    return games["GAME_ID"].unique().tolist()

def fetch_pbp(game_id: str) -> pd.DataFrame:
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
    df = pbp.get_data_frames()[0]
    df["GAME_ID"] = game_id
    return df

def collect(seasons=("2021-22", "2022-23", "2023-24")):
    all_frames = []
    for season in seasons:
        ids = get_season_game_ids(season)
        print(f"{season}: {len(ids)} games")
        for gid in ids:
            cache = OUTPUT / f"{gid}.parquet"
            if cache.exists():
                all_frames.append(pd.read_parquet(cache))
                continue
            try:
                df = fetch_pbp(gid)
                df.to_parquet(cache, index=False)
                all_frames.append(df)
            except Exception as e:
                print(f"  skip {gid}: {e}")
            time.sleep(0.6)   # respect nba_api rate limit
    return pd.concat(all_frames, ignore_index=True)

if __name__ == "__main__":
    df = collect()
    df.to_parquet("data/pbp_raw.parquet", index=False)
    print(f"Saved {len(df):,} play rows")