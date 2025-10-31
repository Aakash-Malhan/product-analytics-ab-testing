import numpy as np, pandas as pd
from scipy import stats

def random_assign_users(users: pd.Series, seed: int = 42, p_treat: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(users)) < p_treat
    return pd.DataFrame({"user_id": users.values, "variant": np.where(mask, "T","C")})

def user_metric(events: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    # per-user metric = #views in first window_days since their t0
    ev = events.copy()
    first_ts = ev.groupby("user_id")["ts"].min().rename("t0")
    ev = ev.merge(first_ts, on="user_id", how="left")
    ev["days_since"] = (ev["ts"] - ev["t0"]).dt.total_seconds() / 86400
    in_win = ev[(ev["event"]=="view") & (ev["days_since"]<=window_days)]
    m = in_win.groupby("user_id").size().rename("views_w{}".format(window_days)).reset_index()
    return m

def srm_check(assignments: pd.DataFrame) -> float:
    # chi-square vs expected 50/50
    nT = (assignments["variant"]=="T").sum()
    nC = (assignments["variant"]=="C").sum()
    total = nT+nC
    expected = np.array([0.5*total, 0.5*total])
    observed = np.array([nT, nC])
    chi2 = ((observed - expected)**2/expected).sum()
    from scipy.stats import chi2 as chi2dist
    p = 1 - chi2dist.cdf(chi2, df=1)
    return float(p)

def cuped_adjust(y: pd.Series, x_cov: pd.Series) -> pd.Series:
    # y = experiment-window metric; x = pre-experiment covariate (e.g., views before t0 or earlier slice)
    theta = (x_cov.cov(y) / (x_cov.var() + 1e-9))
    return y - theta * x_cov

def simulate_treatment_effect(metric_df: pd.DataFrame, assignments: pd.DataFrame, lift_pct: float = 0.12, seed: int = 7):
    # apply synthetic lift to T to emulate feature impact
    df = metric_df.merge(assignments, on="user_id", how="left").fillna("C")
    rng = np.random.default_rng(seed)
    base = df["views_w7"].astype(float).values
    noise = rng.normal(0, 0.5, size=len(df))
    treated = np.where(df["variant"]=="T", base*(1+lift_pct) + noise, base + noise)
    df["y"] = treated
    return df

def ttest_and_ci(df: pd.DataFrame):
    yT = df.loc[df["variant"]=="T","y"].values
    yC = df.loc[df["variant"]=="C","y"].values
    t, p = stats.ttest_ind(yT, yC, equal_var=False)
    lift = (yT.mean() - yC.mean()) / (yC.mean()+1e-9)
    # 95% CI for difference-in-means
    se = np.sqrt(yT.var(ddof=1)/len(yT) + yC.var(ddof=1)/len(yC))
    diff = yT.mean() - yC.mean()
    lo, hi = diff - 1.96*se, diff + 1.96*se
    return {
        "mean_T": float(yT.mean()), "mean_C": float(yC.mean()),
        "diff": float(diff), "lift_pct": float(lift*100),
        "t_stat": float(t), "p_value": float(p),
        "ci_diff_95": (float(lo), float(hi))
    }
