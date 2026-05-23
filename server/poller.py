import time
from nba_api.live.nba.endpoints import playbyplay, boxscore
from features.builder import parse_clock, REGULATION_SECONDS

class LivePoller:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self._last_event = -1

    def get_current_state(self) -> dict | None:
        try:
            bs  = boxscore.BoxScore(game_id=self.game_id)
            pbp = playbyplay.PlayByPlay(game_id=self.game_id)

            bs_data  = bs.get_dict()["game"]
            pbp_acts = pbp.get_dict()["game"]["actions"]

            home = bs_data["homeTeam"]
            away = bs_data["awayTeam"]

            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)

            # Period + clock from the latest action
            latest = pbp_acts[-1] if pbp_acts else {}
            period = int(latest.get("period", 1))
            clock  = latest.get("clock", "PT12M00.00S")
            # Convert ISO 8601 duration PT{m}M{s}S → mm:ss
            import re
            m = re.search(r"PT(\d+)M([\d.]+)S", clock)
            clock_str = f"{int(m.group(1))}:{int(float(m.group(2))):02d}" if m else "12:00"

            secs_left = parse_clock(clock_str, period)
            time_norm = (REGULATION_SECONDS - max(secs_left, 0)) / REGULATION_SECONDS

            home_fouls = sum(
                1 for a in pbp_acts
                if "foul" in str(a.get("actionType", "")).lower()
                and a.get("teamId") == home.get("teamId")
            )
            away_fouls = sum(
                1 for a in pbp_acts
                if "foul" in str(a.get("actionType", "")).lower()
                and a.get("teamId") == away.get("teamId")
            )

            features = {
                "score_diff":        home_score - away_score,
                "secs_left":         secs_left,
                "time_elapsed_norm": time_norm,
                "home_fouls":        home_fouls,
                "away_fouls":        away_fouls,
                "foul_diff":         home_fouls - away_fouls,
                "home_timeouts":     home.get("timeoutsRemaining", 7),
                "away_timeouts":     away.get("timeoutsRemaining", 7),
                "possession_home":   1 if home.get("inBonus") else 0,
                "is_overtime":       1 if period > 4 else 0,
                "period":            period,
            }

            return {
                "game_id":    self.game_id,
                "home_team":  home.get("teamName", "Home"),
                "away_team":  away.get("teamName", "Away"),
                "home_score": home_score,
                "away_score": away_score,
                "period":     period,
                "clock":      clock_str,
                "features":   features,
            }

        except Exception as e:
            print(f"[poller] error: {e}")
            return None