"""
Streamlit Dashboard — §4.E.1

Operational metrics and experiment state visualization.
Entry point: streamlit run services/worker/dashboard.py

Panels:
  - Session overview with metric counts
  - AU12 intensity time-series plot (§11)
  - Acoustic metrics (pitch, jitter, shimmer) charts (§11)
  - Thompson Sampling experiment state (§4.E.1)
"""

from __future__ import annotations

import os
from typing import Any


def _get_db_dsn() -> str:
    """Build PostgreSQL DSN from environment variables (§2 step 7)."""
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'postgres')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('POSTGRES_DB', 'lsie')} "
        f"user={os.environ.get('POSTGRES_USER', 'lsie')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', '')}"
    )


def _query(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a parameterized query and return list of dicts (§2 step 7)."""
    import psycopg2  # Lazy import — container-only dependency

    conn = psycopg2.connect(_get_db_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row, strict=True)) for row in rows]
    finally:
        conn.close()


# --- SQL Queries (§2 step 7 — parameterized) ---

_SESSIONS_SQL = """
    SELECT s.session_id, s.stream_url, s.started_at, s.ended_at,
           COUNT(m.id) AS metric_count
    FROM sessions s
    LEFT JOIN metrics m ON s.session_id = m.session_id
    GROUP BY s.session_id
    ORDER BY s.started_at DESC
"""

_AU12_SQL = """
    SELECT segment_id, timestamp_utc, au12_intensity
    FROM metrics
    WHERE session_id = %(session_id)s AND au12_intensity IS NOT NULL
    ORDER BY timestamp_utc ASC
"""

_ACOUSTIC_SQL = """
    SELECT segment_id, timestamp_utc, pitch_f0, jitter, shimmer
    FROM metrics
    WHERE session_id = %(session_id)s
          AND (pitch_f0 IS NOT NULL OR jitter IS NOT NULL OR shimmer IS NOT NULL)
    ORDER BY timestamp_utc ASC
"""

_EXPERIMENTS_SQL = """
    SELECT experiment_id, arm, alpha_param, beta_param, updated_at
    FROM experiments
    ORDER BY experiment_id, arm
"""


def main() -> None:
    """Render the Streamlit dashboard (§4.E.1)."""
    import streamlit as st  # Lazy import — container-only dependency

    st.set_page_config(page_title="LSIE-MLF Dashboard", layout="wide")
    st.title("LSIE-MLF Operational Dashboard")

    # --- Session Overview ---
    st.header("Session Overview")
    try:
        sessions = _query(_SESSIONS_SQL)
    except Exception as exc:
        st.error(f"Database connection failed: {exc}")
        st.stop()
        return  # unreachable, but keeps mypy happy

    if not sessions:
        st.info("No sessions found. Start a stream to generate data.")
        _render_experiments(st)
        return

    # §11 — Session table with metric counts
    import pandas as pd  # Lazy import — container-only dependency

    sessions_df = pd.DataFrame(sessions)
    st.dataframe(sessions_df, use_container_width=True)

    # Session selector
    session_ids = [s["session_id"] for s in sessions]

    def _fmt_session(sid: str) -> str:
        count = next((s["metric_count"] for s in sessions if s["session_id"] == sid), 0)
        return f"{sid} ({count} metrics)"

    selected_session: str = st.selectbox(
        "Select session for detail view",
        options=session_ids,
        format_func=_fmt_session,
    )

    if selected_session:
        _render_au12(st, pd, selected_session)
        _render_acoustic(st, pd, selected_session)

    _render_experiments(st)


def _render_au12(st: Any, pd: Any, session_id: str) -> None:
    """§11 — AU12 Intensity Score time-series plot."""
    st.header("AU12 Intensity Time-Series")
    au12_data = _query(_AU12_SQL, {"session_id": session_id})

    if not au12_data:
        st.info("No AU12 data available for this session.")
        return

    df = pd.DataFrame(au12_data)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    st.line_chart(df.set_index("timestamp_utc")["au12_intensity"])


def _render_acoustic(st: Any, pd: Any, session_id: str) -> None:
    """§11 — Vocal Pitch, Jitter, Shimmer time-series charts."""
    st.header("Acoustic Metrics Time-Series")
    acoustic_data = _query(_ACOUSTIC_SQL, {"session_id": session_id})

    if not acoustic_data:
        st.info("No acoustic data available for this session.")
        return

    df = pd.DataFrame(acoustic_data)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    ts_df = df.set_index("timestamp_utc")

    # §11 — Separate charts for each acoustic metric
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Pitch F0 (Hz)")
        if "pitch_f0" in ts_df.columns:
            st.line_chart(ts_df["pitch_f0"].dropna())
    with col2:
        st.subheader("Jitter")
        if "jitter" in ts_df.columns:
            st.line_chart(ts_df["jitter"].dropna())
    with col3:
        st.subheader("Shimmer")
        if "shimmer" in ts_df.columns:
            st.line_chart(ts_df["shimmer"].dropna())


def _render_experiments(st: Any) -> None:
    """§4.E.1 — Thompson Sampling experiment state visualization."""
    st.header("Thompson Sampling Experiments")
    try:
        experiments = _query(_EXPERIMENTS_SQL)
    except Exception:
        st.warning("Could not load experiment data.")
        return

    if not experiments:
        st.info("No experiments configured yet.")
        return

    import pandas as pd  # Lazy import — container-only dependency

    df = pd.DataFrame(experiments)

    # §4.E.1 — Display alpha/beta parameters and computed mean
    df["mean"] = df["alpha_param"] / (df["alpha_param"] + df["beta_param"])
    df["variance"] = (df["alpha_param"] * df["beta_param"]) / (
        (df["alpha_param"] + df["beta_param"]) ** 2 * (df["alpha_param"] + df["beta_param"] + 1)
    )

    # Group by experiment_id for display
    for exp_id in df["experiment_id"].unique():
        st.subheader(f"Experiment: {exp_id}")
        exp_df = df[df["experiment_id"] == exp_id][
            ["arm", "alpha_param", "beta_param", "mean", "variance", "updated_at"]
        ]
        st.dataframe(exp_df, use_container_width=True)

        # §4.E.1 — Bar chart of arm means for quick comparison
        st.bar_chart(exp_df.set_index("arm")["mean"])


if __name__ == "__main__":
    main()
