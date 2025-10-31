import numpy as np, pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass

SECONDS_PER_DAY = 86400

@dataclass
class PreparedData:
    events: pd.DataFrame      # columns: user_id, event, item_id, ts (datetime)
    users: pd.DataFrame
    movies: pd.DataFrame

def prepare_events(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame) -> PreparedData:
    df = ratings.copy()
    df.rename(columns={"userId":"user_id","movieId":"item_id"}, inplace=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s")
    # Build engagement events from ratings:
    # every rating -> a "view"; rating>=4 -> also "like" with high prob
    views = df[["user_id","item_id","ts"]].copy()
    views["event"] = "view"
    likes = df[df["rating"] >= 4][["user_id","item_id","ts"]].copy()
    likes["event"] = "like"
    # Optional: simulate "comment" or "share" sparsely
    rng = np.random.default_rng(42)
    maybe_comment = df[df["rating"] >= 4.5][["user_id","item_id","ts"]].copy()
    maybe_comment["event"] = "comment"
    # concat
    events = pd.concat([views, likes, maybe_comment], ignore_index=True)
    events = events.sort_values("ts").reset_index(drop=True)
    # normalize types
    events["user_id"] = events["user_id"].astype(int)
    events["item_id"] = events["item_id"].astype(int)
    users = users.rename(columns={"userId":"user_id"}).copy()
    movies = movies.rename(columns={"movieId":"item_id"}).copy()
    return PreparedData(events=events, users=users, movies=movies)

def build_cohorts(events: pd.DataFrame, period: str = "W") -> pd.DataFrame:
    # cohort = user's first event period; retention = active in subsequent periods
    ev = events.copy()
    ev["period"] = ev["ts"].dt.to_period(period).dt.start_time
    first = ev.groupby("user_id")["period"].min().rename("cohort")
    ev = ev.merge(first, on="user_id", how="left")
    retention = (ev
        .groupby(["cohort","period"])["user_id"].nunique()
        .rename("active_users")
        .reset_index())
    # cohort sizes
    sizes = retention[retention["period"] == retention["cohort"]][["cohort","active_users"]].rename(columns={"active_users":"cohort_size"})
    out = retention.merge(sizes, on="cohort", how="left")
    out["retention_rate"] = out["active_users"] / out["cohort_size"]
    return out.sort_values(["cohort","period"])

def build_funnel(events: pd.DataFrame) -> pd.DataFrame:
    # Simple funnel over first 7 days from cohort start:
    # Step1: any view   Step2: >=5 views in 3d ("activation")  Step3: return in day 7 ("week1_retention")
    ev = events.copy()
    first_ts = ev.groupby("user_id")["ts"].min().rename("t0")
    ev = ev.merge(first_ts, on="user_id", how="left")
    ev["days_since"] = (ev["ts"] - ev["t0"]).dt.total_seconds() / SECONDS_PER_DAY

    users = ev[["user_id","t0"]].drop_duplicates()
    step1 = users.assign(step1=1)

    activated = (ev[(ev["event"]=="view") & (ev["days_since"]<=3.0)]
                 .groupby("user_id").size().rename("views_3d").reset_index())
    step2_users = activated[activated["views_3d"]>=5][["user_id"]].assign(step2=1)

    week1 = (ev[(ev["days_since"]>=7.0) & (ev["days_since"]<8.0)]
             .groupby("user_id").size().rename("events_day7").reset_index())
    step3_users = week1[week1["events_day7"]>0][["user_id"]].assign(step3=1)

    df = users.merge(step1, on="user_id", how="left") \
              .merge(step2_users, on="user_id", how="left") \
              .merge(step3_users, on="user_id", how="left") \
              .fillna(0)

    totals = pd.DataFrame({
        "step": ["signup","activation(5 views in 3d)","day7_retention"],
        "users": [df["step1"].sum(), df["step2"].sum(), df["step3"].sum()]
    })
    totals["rate_vs_signup"] = totals["users"] / totals["users"].iloc[0]
    return totals

def product_kpis(events: pd.DataFrame) -> Dict[str, float]:
    ev = events.copy()
    ev["date"] = ev["ts"].dt.date
    dau = ev.groupby("date")["user_id"].nunique().rename("DAU").to_frame()
    dau["MAU_rolling"] = ev.set_index("ts")["user_id"].groupby(pd.Grouper(freq="D")).nunique().rolling(30, min_periods=1).mean().values
    kpis = {
        "avg_DAU": float(dau["DAU"].mean()),
        "peak_DAU": float(dau["DAU"].max()),
        "DAU/MAU_proxy": float(dau["DAU"].mean() / (dau["MAU_rolling"].mean() + 1e-9)),
        "avg_daily_events": float(ev.groupby("date").size().mean())
    }
    return kpis
