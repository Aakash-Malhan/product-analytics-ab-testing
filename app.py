from pathlib import Path
import gradio as gr
import pandas as pd
import plotly.express as px

from data_download import download_movielens, DATA_DIR
from analytics import (
    prepare_events,
    build_cohorts,
    build_funnel,
    product_kpis,
)
from ab_test import (
    random_assign_users,
    user_metric,
    simulate_treatment_effect,
    ttest_and_ci,
    srm_check,
    cuped_adjust,
)

# -----------------------
# Helpers
# -----------------------
def _read_uploaded_csv(upload) -> pd.DataFrame:
    """Read a Gradio-uploaded CSV (utf-8 → latin-1 fallback)."""
    def _read_any(path_like):
        p = str(path_like)
        try:
            return pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(p, encoding="latin-1")
    if upload is None:
        raise ValueError("No file provided")
    if isinstance(upload, (str, Path)):
        return _read_any(upload)
    if hasattr(upload, "name"):
        return _read_any(upload.name)
    raise ValueError("Unsupported upload type; please upload a CSV file.")

def ensure_data():
    """Ensure MovieLens CSVs exist; return (ratings, users, movies, err)."""
    try:
        download_movielens()
        ratings = pd.read_csv(DATA_DIR / "ratings.csv")
        users   = pd.read_csv(DATA_DIR / "users.csv")
        movies  = pd.read_csv(DATA_DIR / "movies.csv")
        return ratings, users, movies, None
    except Exception as e:
        return None, None, None, f"Download/parse failed: {e}"

# -----------------------
# Build analytics views
# -----------------------
def build_all(upload: gr.File | None):
    try:
        if upload is not None:
            ratings = _read_uploaded_csv(upload)
            users = pd.DataFrame({"userId": ratings["userId"].unique()})
            movies = pd.DataFrame({"movieId": ratings["movieId"].unique()})
            err = None
        else:
            ratings, users, movies, err = ensure_data()
        if err:
            return None, None, None, None, gr.update(value=f"❌ {err}")

        pdata = prepare_events(ratings, users, movies)
        kpis = product_kpis(pdata.events)

        cohorts = build_cohorts(pdata.events, period="W")
        cohort_fig = px.line(
            cohorts, x="period", y="retention_rate", color="cohort",
            title="Weekly Retention by Cohort"
        )

        funnel = build_funnel(pdata.events)
        funnel_fig = px.bar(
            funnel, x="step", y="users", text="users",
            title="Activation Funnel (first 7 days)"
        )

        kpi_txt = (
            f"**Avg DAU:** {kpis['avg_DAU']:.0f} | "
            f"**Peak DAU:** {kpis['peak_DAU']:.0f} | "
            f"**DAU/MAU (proxy):** {kpis['DAU/MAU_proxy']:.2f} | "
            f"**Avg Daily Events:** {kpis['avg_daily_events']:.0f}"
        )

        return pdata.events.head(1000), cohort_fig, funnel_fig, kpi_txt, gr.update(value="✅ Data ready.")
    except Exception as e:
        return None, None, None, None, gr.update(value=f"❌ Error: {e}")

# -----------------------
# A/B simulator (final fix: no duplicate 'y')
# -----------------------
def run_experiment(lift_pct, seed):
    try:
        # 1) Load data & events
        ratings, users, movies, err = ensure_data()
        if err:
            return f"❌ {err}", None, pd.DataFrame()
        pdata = prepare_events(ratings, users, movies)

        # 2) Variant assignment
        assignments = random_assign_users(
            pdata.events["user_id"].drop_duplicates(), seed=int(seed), p_treat=0.5
        )
        p_srm = srm_check(assignments)

        # 3) Metrics (7-day views + 1-day covariate)
        m7 = user_metric(pdata.events, window_days=7)                     # -> views_w7
        m1 = user_metric(pdata.events, window_days=1).rename(columns={"views_w1": "pre_views"})
        m = m7.merge(m1, on="user_id", how="left").fillna(0)

        for c in ("views_w7", "pre_views"):
            if c in m.columns:
                m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0.0)
        if m.empty or "views_w7" not in m.columns:
            return "⚠️ Not enough per-user data to compute 7-day views.", None, pd.DataFrame()

        # 4) Simulate treatment
        df = simulate_treatment_effect(m, assignments, lift_pct=float(lift_pct), seed=int(seed))

        # Guarantee required cols
        if "variant" not in df.columns:
            df = df.merge(assignments, on="user_id", how="left")
        df["variant"] = df["variant"].fillna("C").astype(str)
        if "y" not in df.columns:
            df["y"] = pd.to_numeric(df.get("views_w7", 0), errors="coerce").fillna(0.0)

        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
        df["pre_views"] = pd.to_numeric(df.get("pre_views", 0), errors="coerce").fillna(0.0)

        # 5) Plain stats — use a trimmed DF with exactly one 'y'
        df_plain = df[["user_id", "variant", "y"]].copy()
        res_plain = ttest_and_ci(df_plain)

        # 6) CUPED stats — build a separate DF (no duplicate 'y')
        df_cuped = df[["user_id", "variant", "pre_views", "y"]].copy()
        df_cuped["y"] = cuped_adjust(df_cuped["y"], df_cuped["pre_views"])
        df_cuped = df_cuped[["user_id", "variant", "y"]]  # only one 'y'
        res_cuped = ttest_and_ci(df_cuped)

        def pretty(res, tag):
            return (
                f"**{tag}** | mean_T: {res['mean_T']:.2f} | mean_C: {res['mean_C']:.2f} | "
                f"diff: {res['diff']:.2f} | lift: {res['lift_pct']:.2f}% | "
                f"t={res['t_stat']:.2f}, p={res['p_value']:.4f}, "
                f"95% CI: [{res['ci_diff_95'][0]:.2f}, {res['ci_diff_95'][1]:.2f}]"
            )

        debug = (
            f"\n\n<small>rows={len(df)}, cols={list(df.columns)[:12]}..., "
            f"SRM p={p_srm:.3f}</small>"
        )
        summary = (
            f"**SRM p-value:** {p_srm:.3f} (flag SRM if < 0.01)\n\n"
            + pretty(res_plain, "Naive T-test")
            + "\n"
            + pretty(res_cuped, "CUPED-adjusted")
            + debug
        )

        # 7) Plot on the plain metric
        fig = px.histogram(
            df_plain, x="y", color="variant", nbins=40, barmode="overlay", opacity=0.6,
            title="User Metric Distribution (views in first 7d)"
        )
        return summary, fig, df_plain.head(20)

    except Exception as e:
        return f"❌ Exception in simulation: {e}", None, pd.DataFrame()

# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks(title="Product Analytics • Cohorts • A/B • CUPED") as demo:
    gr.Markdown("# 📊 Product Analytics & Experimentation (Meta-style)")
    gr.Markdown(
        "Cohorts, activation funnel, KPIs, and an A/B test simulator with CUPED and SRM checks. "
        "Upload your own `ratings.csv`-like table (userId,movieId,rating,timestamp) **or** use MovieLens 1M."
    )

    with gr.Row():
        upload = gr.File(label="Upload ratings.csv (optional)")
        build_btn = gr.Button("Build from Dataset", variant="primary")
    status = gr.Markdown()

    events_df = gr.Dataframe(label="Sample Events (head)")
    cohort_plot = gr.Plot(label="Retention by Cohort")
    funnel_plot = gr.Plot(label="Activation Funnel")
    kpis_md = gr.Markdown()

    build_btn.click(build_all, inputs=[upload],
                    outputs=[events_df, cohort_plot, funnel_plot, kpis_md, status])

    gr.Markdown("---")
    gr.Markdown("## 🧪 A/B Test Simulator (with CUPED)")
    with gr.Row():
        lift = gr.Slider(0.00, 0.30, value=0.12, step=0.01,
                         label="Simulated Treatment Lift on 7-day views")
        seed = gr.Slider(1, 9999, value=7, step=1, label="Random Seed")
        run_btn = gr.Button("Run Simulation", variant="primary")
    ab_summary = gr.Markdown()
    ab_plot = gr.Plot()
    ab_table = gr.Dataframe(label="Sample per-user metrics")

    run_btn.click(run_experiment, inputs=[lift, seed],
                  outputs=[ab_summary, ab_plot, ab_table])

if __name__ == "__main__":
    demo.launch()
