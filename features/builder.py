import pandas as pd, numpy as np, re

REGULATION_SECONDS = 48 * 60          # 2880
PERIOD_SECONDS     = 12 * 60          # 720
OT_SECONDS         = 5  * 60          # 300

def parse_clock(clock_str: str, period: int) -> float:
    """Return seconds remaining in regulation (negative = overtime)."""
    if not isinstance(clock_str, str):
        return 0.0
    m = re.match(r"(\d+):(\d+)", clock_str)
    if not m:
        return 0.0
    mins, secs = int(m.group(1)), int(m.group(2))
    elapsed_in_period = PERIOD_SECONDS - (mins * 60 + secs)
    if period <= 4:
        elapsed_total = (period - 1) * PERIOD_SECONDS + elapsed_in_period
        return REGULATION_SECONDS - elapsed_total
    else:
        ot_elapsed = (period - 5) * OT_SECONDS + elapsed_in_period
        return -(ot_elapsed)           # negative means OT

def score_from_description(desc: str) -> int:
    """Extract point value from a play description string."""
    desc = str(desc).upper()
    if "3PT" in desc or "THREE" in desc:
        return 3
    if "FREE THROW" in desc or "FT" in desc:
        return 1
    if any(x in desc for x in ["DUNK","LAYUP","ALLEY","SHOT","JUMPER","HOOK"]):
        return 2
    return 0

def build_features(pbp_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_id, game in pbp_raw.groupby("GAME_ID"):
        game = game.sort_values("EVENTNUM").reset_index(drop=True)

        home_score, away_score = 0, 0
        home_fouls, away_fouls  = 0, 0
        home_timeouts, away_timeouts = 7, 7   # standard NBA allotment
        possession = "home"   # simplified; refine with HOMEDESCRIPTION/VISITORDESCRIPTION

        for _, row in game.iterrows():
            period  = int(row.get("PERIOD", 1))
            clock   = str(row.get("PCTIMESTRING", "12:00"))
            secs_left = parse_clock(clock, period)

            home_desc = str(row.get("HOMEDESCRIPTION", "") or "")
            away_desc = str(row.get("VISITORDESCRIPTION", "") or "")

            # Score tracking
            home_score += score_from_description(home_desc)
            away_score += score_from_description(away_desc)

            # Foul tracking
            if "FOUL" in home_desc.upper():
                home_fouls += 1
            if "FOUL" in away_desc.upper():
                away_fouls += 1

            # Timeout tracking
            if "TIMEOUT" in home_desc.upper() and home_timeouts > 0:
                home_timeouts -= 1
            if "TIMEOUT" in away_desc.upper() and away_timeouts > 0:
                away_timeouts -= 1

            # Possession heuristic
            if home_desc and not away_desc:
                possession = "home"
            elif away_desc and not home_desc:
                possession = "away"

            # Terminal state: did home team win?
            # We tag this per-row; home_win is computed after the loop
            rows.append({
                "game_id":        game_id,
                "event_num":      row.get("EVENTNUM"),
                "period":         period,
                "secs_left":      secs_left,
                "score_diff":     home_score - away_score,   # +ve = home leading
                "home_score":     home_score,
                "away_score":     away_score,
                "home_fouls":     home_fouls,
                "away_fouls":     away_fouls,
                "foul_diff":      home_fouls - away_fouls,
                "home_timeouts":  home_timeouts,
                "away_timeouts":  away_timeouts,
                "possession_home": 1 if possession == "home" else 0,
                "is_overtime":    1 if period > 4 else 0,
            })

        # Tag the final outcome for all rows in this game
        outcome = 1 if home_score > away_score else 0
        for r in rows[-len(game):]:
            r["home_win"] = outcome

    df = pd.DataFrame(rows)
    # Normalise time: 0 = tip-off, 1 = end of regulation
    df["time_elapsed_norm"] = (REGULATION_SECONDS - df["secs_left"].clip(0)) / REGULATION_SECONDS
    return df.dropna(subset=["home_win"])

if __name__ == "__main__":
    raw = pd.read_parquet("data/pbp_raw.parquet")
    features = build_features(raw)
    features.to_parquet("data/features.parquet", index=False)
    print(features[["score_diff","secs_left","home_fouls","home_win"]].describe())