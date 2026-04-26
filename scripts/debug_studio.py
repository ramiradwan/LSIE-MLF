#!/usr/bin/env python3  
from __future__ import annotations  
  
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any
  
import cv2  
import mediapipe as mp  
import numpy as np  
  
try:  
    import psutil  # Optional  
except ImportError:  
    psutil = None  
  
from PySide6.QtCore import QEvent, QRectF, QSize, Qt, QThread, Signal  
from PySide6.QtGui import (  
    QColor,  
    QCloseEvent,  
    QImage,  
    QKeySequence,  
    QLinearGradient,  
    QPainter,  
    QPainterPath,  
    QPen,  
    QPixmap,  
    QShortcut,  
)  
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
  
# -----------------------------------------------------------------------------
# LSIE-MLF Debug Studio (§4.E.1 operator tooling)
#
# Data consumption contract — three disjoint read paths, one write path.
# Each worker thread below owns exactly one path; no thread mixes sources.
#
#   1. LIVE PATH  (StreamThread)
#      Source:      /tmp/ipc/video_stream.mkv (IPC Pipe, in-process MediaPipe)
#      Cadence:     ~30 fps, sub-100 ms frame-to-display latency
#      Why direct:  Per-frame landmarks are never persisted (§5 data governance
#                   — only per-segment aggregates enter the Persistent Store).
#                   A DB or API hop would add poll latency that would break the
#                   live preview; the data also does not exist at those layers.
#      Consumers:   VideoPane, AU12Tile, TrackingPointsTile, Face/FPS tiles.
#
#   2. ANALYTICS PATH  (AnalyticsThread, polls OPERATOR_SNAPSHOT_SQL)
#      Source:      PostgreSQL via `docker compose exec postgres psql`
#      Cadence:     ANALYTICS_POLL_SECONDS (default 1 s), fingerprint-deduped
#      Why direct:  Operator readback is a high-frequency multi-table JOIN over
#                   session history. Going through FastAPI per tick would cost
#                   an HTTP round-trip for every poll and force the API to
#                   grow a read endpoint for every table we surface. Direct
#                   psql keeps the GUI decoupled from the API uptime and
#                   consistent across tabs. The CLI (scripts/lsie_cli.py) has
#                   the opposite tradeoff — one-shot queries, so it uses the
#                   REST API.
#      Consumers:   All "Analytics" tab panels and the right-hand side panels
#                   on the "Live" tab (transcript, semantic eval, reward,
#                   acoustic metrics, physiology, co-modulation, per-arm).
#
#   3. HARDWARE PATH  (HardwareTelemetryThread)
#      Source:      nvidia-smi (host) + `adb shell` (device), out-of-band
#      Why direct:  Host/device telemetry is not a Persistent Store concern;
#                   it never transits the API or DB. Read straight from the
#                   tools that own it.
#      Consumers:   Host System tile, Device CPU tile, Foreground App tile.
#
#   WRITE PATH  (stimulus inject)
#      Target:      FastAPI `POST /api/v1/stimulus`
#      Why API:     The API owns the Redis pub/sub publish. Write authority
#                   stays with a single service — the GUI does not talk to
#                   Redis directly.
# -----------------------------------------------------------------------------
  
APP_NAME = "LSIE-MLF Debug Studio"  
  
cv2.setUseOptimized(True)  
try:  
    cv2.setNumThreads(1)  
except Exception:  
    pass  
  
COMPOSE_COMMAND = shlex.split(os.getenv("LSIE_COMPOSE_COMMAND", "docker compose")) or [  
    "docker",  
    "compose",  
]  
STREAM_SERVICE = os.getenv("LSIE_STREAM_SERVICE", "stream_scrcpy")  
VIDEO_PIPE = os.getenv("LSIE_VIDEO_PIPE", "/tmp/ipc/video_stream.mkv")  
  
FRAME_WIDTH = max(1, int(os.getenv("LSIE_DEBUG_FRAME_WIDTH", "540")))  
FRAME_HEIGHT = max(1, int(os.getenv("LSIE_DEBUG_FRAME_HEIGHT", "960")))  
OUTPUT_FPS = max(1, int(os.getenv("LSIE_DEBUG_FPS", "30")))  
  
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3  
FRAME_ASPECT = FRAME_WIDTH / max(FRAME_HEIGHT, 1)  
  
OVERLAY_RAW = "raw"  
OVERLAY_LIPS = "lips"  
OVERLAY_FULL = "full"  
  
POSTGRES_SERVICE = os.getenv("LSIE_DB_SERVICE", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "lsie")
POSTGRES_DB = os.getenv("POSTGRES_DB", "lsie_mlf")
ANALYTICS_POLL_SECONDS = max(
    0.25, float(os.getenv("LSIE_ANALYTICS_POLL_SECONDS", "1.0"))
)
API_BASE = os.getenv("LSIE_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_EXPERIMENT_ID = os.getenv("LSIE_EXPERIMENT_ID", "greeting_line_v1")
METRICS_UPDATE_INTERVAL = 0.15  
  
MODE_LABELS = {  
    OVERLAY_RAW: "Raw",  
    OVERLAY_LIPS: "Lip Points",  
    OVERLAY_FULL: "Face Mesh",  
}  
  
LIP_MARKER_COLOR = (0, 255, 0)  
LIP_OUTLINE_COLOR = (0, 0, 0)  
  
# Core snapshot — queries only tables that exist in every v2+ deployment.
# Split from the physiology/comodulation query so a missing v3.1 migration
# on an old volume doesn't break the rest of the operator readback.
OPERATOR_SNAPSHOT_SQL = r"""
WITH recent_metrics AS (
  SELECT *
  FROM metrics
  ORDER BY timestamp_utc DESC
  LIMIT 60
),
latest_transcript AS (
  SELECT *
  FROM transcripts
  ORDER BY timestamp_utc DESC
  LIMIT 1
),
latest_evaluation AS (
  SELECT *
  FROM evaluations
  ORDER BY timestamp_utc DESC
  LIMIT 1
),
latest_encounter AS (
  SELECT *
  FROM encounter_log
  ORDER BY created_at DESC
  LIMIT 1
),
greeting_experiments AS (
  SELECT *
  FROM experiments
  WHERE experiment_id = 'greeting_line_v1'
  ORDER BY arm
),
active_session AS (
  SELECT session_id
  FROM sessions
  ORDER BY started_at DESC
  LIMIT 1
),
encounter_summary AS (
  SELECT
    arm,
    COUNT(*) AS encounter_count,
    COUNT(*) FILTER (WHERE is_valid) AS valid_count,
    AVG(gated_reward) AS avg_reward,
    AVG(gated_reward) FILTER (WHERE is_valid) AS avg_valid_reward,
    AVG(semantic_gate::double precision) AS gate_rate,
    AVG(n_frames::double precision) AS avg_frames
  FROM encounter_log
  WHERE experiment_id = 'greeting_line_v1'
  GROUP BY arm
  ORDER BY arm
)
SELECT json_build_object(
  'metrics',
  COALESCE(
    (SELECT row_to_json(m) FROM (
      SELECT *
      FROM recent_metrics
      ORDER BY timestamp_utc DESC
      LIMIT 1
    ) m),
    '{}'::json
  ),
  'metrics_history',
  COALESCE(
    (SELECT json_agg(row_to_json(mh)) FROM (
      SELECT
        to_jsonb(rm) -> 'pitch_f0' AS pitch_f0,
        to_jsonb(rm) -> 'jitter' AS jitter,
        to_jsonb(rm) -> 'shimmer' AS shimmer,
        to_jsonb(rm) -> 'f0_valid_measure' AS f0_valid_measure,
        to_jsonb(rm) -> 'f0_valid_baseline' AS f0_valid_baseline,
        to_jsonb(rm) -> 'perturbation_valid_measure' AS perturbation_valid_measure,
        to_jsonb(rm) -> 'perturbation_valid_baseline' AS perturbation_valid_baseline,
        to_jsonb(rm) -> 'voiced_coverage_measure_s' AS voiced_coverage_measure_s,
        to_jsonb(rm) -> 'voiced_coverage_baseline_s' AS voiced_coverage_baseline_s,
        to_jsonb(rm) -> 'f0_mean_measure_hz' AS f0_mean_measure_hz,
        to_jsonb(rm) -> 'f0_mean_baseline_hz' AS f0_mean_baseline_hz,
        to_jsonb(rm) -> 'f0_delta_semitones' AS f0_delta_semitones,
        to_jsonb(rm) -> 'jitter_mean_measure' AS jitter_mean_measure,
        to_jsonb(rm) -> 'jitter_mean_baseline' AS jitter_mean_baseline,
        to_jsonb(rm) -> 'jitter_delta' AS jitter_delta,
        to_jsonb(rm) -> 'shimmer_mean_measure' AS shimmer_mean_measure,
        to_jsonb(rm) -> 'shimmer_mean_baseline' AS shimmer_mean_baseline,
        to_jsonb(rm) -> 'shimmer_delta' AS shimmer_delta
      FROM recent_metrics rm
      ORDER BY rm.timestamp_utc DESC
    ) mh),
    '[]'::json
  ),
  'transcript',
  COALESCE((SELECT row_to_json(t) FROM latest_transcript t), '{}'::json),
  'evaluation',
  COALESCE((SELECT row_to_json(e) FROM latest_evaluation e), '{}'::json),
  'encounter',
  COALESCE((SELECT row_to_json(el) FROM latest_encounter el), '{}'::json),
  'experiments',
  COALESCE((SELECT json_agg(x) FROM greeting_experiments x), '[]'::json),
  'session_id',
  (SELECT session_id::text FROM active_session),
  'encounter_summary',
  COALESCE((SELECT json_agg(row_to_json(es)) FROM encounter_summary es), '[]'::json)
);
"""

# Physiology snapshot — depends on v3.1 tables (data/sql/03-physiology.sql).
# If those tables don't exist yet, AnalyticsThread degrades gracefully: the
# core query still flows, and this one is skipped until the next restart.
OPERATOR_PHYSIO_SQL = r"""
WITH active_session AS (
  SELECT session_id
  FROM sessions
  ORDER BY started_at DESC
  LIMIT 1
),
latest_streamer_physio AS (
  SELECT *
  FROM physiology_log
  WHERE session_id = (SELECT session_id FROM active_session)
    AND subject_role = 'streamer'
  ORDER BY created_at DESC
  LIMIT 1
),
latest_operator_physio AS (
  SELECT *
  FROM physiology_log
  WHERE session_id = (SELECT session_id FROM active_session)
    AND subject_role = 'operator'
  ORDER BY created_at DESC
  LIMIT 1
),
comodulation_history AS (
  SELECT *
  FROM comodulation_log
  WHERE session_id = (SELECT session_id FROM active_session)
  ORDER BY window_end_utc DESC
  LIMIT 60
)
SELECT json_build_object(
  'streamer_physio',
  COALESCE((SELECT row_to_json(s) FROM latest_streamer_physio s), '{}'::json),
  'operator_physio',
  COALESCE((SELECT row_to_json(o) FROM latest_operator_physio o), '{}'::json),
  'comodulation',
  COALESCE(
    (SELECT json_agg(row_to_json(c) ORDER BY c.window_end_utc ASC) FROM (
      SELECT window_end_utc, window_minutes, co_modulation_index,
             n_paired_observations, coverage_ratio,
             streamer_rmssd_mean, operator_rmssd_mean
      FROM comodulation_history
    ) c),
    '[]'::json
  )
);
"""
  
APP_STYLESHEET = """  
QWidget {  
    color: #F8FAFC;  
    font-family: Inter, "Segoe UI", Arial, sans-serif;  
    font-size: 14px;  
}  
QMainWindow {  
    background: transparent;  
}  
QFrame#MainFrame {  
    background: rgba(11, 15, 22, 0.98);  
    border: 1px solid rgba(255, 255, 255, 0.08);  
    border-radius: 12px;  
}  
QFrame#MainFrame[maximized="true"] {  
    border-radius: 0px;  
    border: none;  
}  
QWidget#TitleBar {  
    background: transparent;  
}  
QPushButton#SysButton {  
    background: transparent;  
    border: none;  
    border-radius: 0px;  
    color: #94A3B8;  
    font-size: 14px;  
}  
QPushButton#SysButton:hover {  
    background: rgba(255, 255, 255, 0.1);  
    color: #FFFFFF;  
}  
QPushButton#SysCloseButton {  
    background: transparent;  
    border: none;  
    border-radius: 0px;  
    border-top-right-radius: 11px;  
    color: #94A3B8;  
    font-size: 14px;  
}  
QFrame#MainFrame[maximized="true"] QPushButton#SysCloseButton {  
    border-top-right-radius: 0px;  
}  
QPushButton#SysCloseButton:hover {  
    background: #E11D48;  
    color: #FFFFFF;  
}  
QFrame#Surface {  
    background: rgba(255, 255, 255, 0.03);  
    border: 1px solid rgba(255, 255, 255, 0.05);  
    border-radius: 16px;  
}  
QFrame#Card {  
    background: rgba(0, 0, 0, 0.2);  
    border: 1px solid rgba(255, 255, 255, 0.04);  
    border-radius: 12px;  
}  
QFrame#InlineStat {  
    background: rgba(255, 255, 255, 0.02);  
    border: 1px solid rgba(255, 255, 255, 0.04);  
    border-radius: 10px;  
}  
QLabel#Title {  
    font-size: 28px;  
    font-weight: 800;  
    color: #FFFFFF;  
    letter-spacing: -0.5px;  
}  
QLabel#Subtitle {  
    font-size: 13px;  
    color: #94A3B8;  
}  
QLabel#PanelTitle {  
    font-size: 16px;  
    font-weight: 600;  
    color: #E2E8F0;  
}  
QLabel#CardTitle {  
    font-size: 11px;  
    font-weight: 700;  
    color: #64748B;  
    text-transform: uppercase;  
    letter-spacing: 0.8px;  
}  
QLabel#CardValue {  
    font-size: 22px;  
    font-weight: 700;  
    color: #FFFFFF;  
}  
QLabel#SecondaryValue {  
    font-size: 12px;  
    font-weight: 600;  
    color: #CBD5E1;  
}  
QLabel#InlineTitle {  
    font-size: 10px;  
    font-weight: 700;  
    color: #64748B;  
    text-transform: uppercase;  
    letter-spacing: 0.8px;  
}  
QLabel#InlineValue {  
    font-size: 18px;  
    font-weight: 700;  
    color: #FFFFFF;  
}  
QLabel#InlineValueCompact {  
    font-size: 13px;  
    font-weight: 600;  
    color: #E2E8F0;  
}  
QLabel#CardHint {  
    font-size: 12px;  
    color: #64748B;  
}  
QLabel#InfoRow {  
    font-size: 12px;  
    color: #CBD5E1;  
}  
QPushButton#GlassButton {  
    background: rgba(255, 255, 255, 0.05);  
    border: 1px solid rgba(255, 255, 255, 0.1);  
    border-radius: 8px;  
    color: #F8FAFC;  
    padding: 8px 16px;  
    font-weight: 600;  
}  
QPushButton#GlassButton:hover {  
    background: rgba(255, 255, 255, 0.1);  
    border: 1px solid rgba(255, 255, 255, 0.15);  
}  
QPushButton#GlassButton:pressed {  
    background: rgba(255, 255, 255, 0.02);  
}  
QFrame#SegmentedControl {  
    background: rgba(0, 0, 0, 0.25);  
    border-radius: 8px;  
    padding: 4px;  
    border: 1px solid rgba(255, 255, 255, 0.04);  
}  
QPushButton[modeButton="true"] {  
    background: transparent;  
    border: none;  
    border-radius: 6px;  
    padding: 6px 18px;  
    color: #64748B;  
    font-weight: 600;  
}  
QPushButton[modeButton="true"]:hover {  
    color: #E2E8F0;  
}  
QPushButton[modeButton="true"]:checked {  
    background: rgba(255, 255, 255, 0.12);  
    color: #FFFFFF;  
}  
QPlainTextEdit {  
    background: transparent;  
    border: none;  
    padding: 8px;  
    color: #CBD5E1;  
    selection-background-color: rgba(59, 130, 246, 0.5);  
    font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;  
    font-size: 12px;  
}  
QPlainTextEdit#TextBox {  
    background: rgba(0, 0, 0, 0.15);  
    border: 1px solid rgba(255, 255, 255, 0.04);  
    border-radius: 8px;  
    padding: 12px;  
    color: #FFFFFF;  
    font-family: Inter, "Segoe UI", Arial, sans-serif;  
    font-size: 15px;  
}  
QScrollArea {  
    background: transparent;  
    border: none;  
}  
QScrollArea > QWidget > QWidget {  
    background: transparent;  
}  
QScrollBar:vertical {  
    border: none;  
    background: transparent;  
    width: 10px;  
    margin: 0px;  
}  
QScrollBar::handle:vertical {  
    background: rgba(255, 255, 255, 0.15);  
    border-radius: 5px;  
    min-height: 20px;  
}  
QScrollBar::handle:vertical:hover {  
    background: rgba(255, 255, 255, 0.25);  
}  
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {  
    height: 0px;  
}  
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {  
    background: transparent;  
}  
QScrollBar:horizontal {  
    border: none;  
    background: transparent;  
    height: 10px;  
    margin: 0px;  
}  
QScrollBar::handle:horizontal {  
    background: rgba(255, 255, 255, 0.15);  
    border-radius: 5px;  
    min-width: 20px;  
}  
QScrollBar::handle:horizontal:hover {  
    background: rgba(255, 255, 255, 0.25);  
}  
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {  
    width: 0px;  
}  
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {  
    background: transparent;  
}  
QSplitter::handle {
    background: transparent;
}
QSplitter::handle:vertical {
    height: 16px;
}
QSplitter::handle:horizontal {
    width: 16px;
}
QTabWidget::pane {
    background: transparent;
    border: none;
    top: -1px;
}
QTabWidget#MainTabs::tab-bar {
    left: 0px;
}
QTabBar {
    background: transparent;
    qproperty-drawBase: 0;
}
QTabBar::tab {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    color: #94A3B8;
    padding: 8px 22px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.3px;
}
QTabBar::tab:hover {
    color: #E2E8F0;
    background: rgba(255, 255, 255, 0.06);
}
QTabBar::tab:selected {
    background: rgba(255, 255, 255, 0.08);
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-bottom: none;
}
QTableWidget {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    color: #E2E8F0;
    gridline-color: rgba(255, 255, 255, 0.04);
    selection-background-color: transparent;
    font-size: 13px;
}
QTableWidget::item {
    padding: 6px 8px;
    border: none;
}
QHeaderView {
    background: transparent;
}
QHeaderView::section {
    background: rgba(255, 255, 255, 0.04);
    color: #94A3B8;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
QTableCornerButton::section {
    background: rgba(255, 255, 255, 0.04);
    border: none;
}
"""
  
  
def run_psql_json(query: str) -> dict[str, Any]:  
    cmd = [  
        *COMPOSE_COMMAND,  
        "exec",  
        "-T",  
        POSTGRES_SERVICE,  
        "psql",  
        "-X",  
        "-U",  
        POSTGRES_USER,  
        "-d",  
        POSTGRES_DB,  
        "-v",  
        "ON_ERROR_STOP=1",  
        "-t",  
        "-A",  
        "-c",  
        query,  
    ]  
    result = subprocess.run(  
        cmd,  
        capture_output=True,  
        text=True,  
        check=False,  
        timeout=6.0,  
    )  
    if result.returncode != 0:  
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "psql failed")  
  
    raw = result.stdout.strip()  
    if not raw:  
        return {}  
  
    try:  
        return json.loads(raw)  
    except json.JSONDecodeError:  
        return json.loads(raw.splitlines()[-1])  
  
  
def pick(mapping: dict[str, Any], *keys: str, default: Any = "--") -> Any:  
    for key in keys:  
        if key in mapping and mapping[key] is not None:  
            return mapping[key]  
    return default  
  
  
def deep_find(value: Any, target_key: str, default: Any = "--") -> Any:  
    if isinstance(value, dict):  
        if target_key in value and value[target_key] is not None:  
            return value[target_key]  
        for nested in value.values():  
            found = deep_find(nested, target_key, default=None)  
            if found is not None:  
                return found  
    elif isinstance(value, list):  
        for nested in value:  
            found = deep_find(nested, target_key, default=None)  
            if found is not None:  
                return found  
    return default if default is not None else None  
  
  
def mode_label(mode: str) -> str:  
    return MODE_LABELS.get(mode, mode.title())  
  
  
def format_optional_float(value: Any, precision: int, default: str = "--") -> str:  
    if value in (None, "", "--"):  
        return default  
    try:  
        number = float(value)  
    except (TypeError, ValueError):  
        return default  
    if not math.isfinite(number):  
        return default  
    return f"{number:.{precision}f}"  
  
  
def is_missing_value(value: Any) -> bool:
    return value is None or value == "" or value == "--"


def first_present(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = mapping.get(key)
        if not is_missing_value(value):
            return value
    return default


def format_delta_line(value: Any, precision: int, suffix: str = "") -> str:
    if is_missing_value(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    sign = "+" if number > 0 else ""
    return f"Δ {sign}{number:.{precision}f}{suffix}"


def format_validity_flag(value: Any) -> str:
    if is_missing_value(value):
        return "--"
    if isinstance(value, bool):
        return "YES" if value else "NO"

    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return "YES"
    if lowered in {"false", "f", "0", "no", "n"}:
        return "NO"
    return text or "--"


def format_validity_pair(
    mapping: dict[str, Any], measure_key: str, baseline_key: str
) -> str:
    if not isinstance(mapping, dict):
        return "--"
    measure = mapping.get(measure_key)
    baseline = mapping.get(baseline_key)
    if is_missing_value(measure) and is_missing_value(baseline):
        return "--"
    return f"M {format_validity_flag(measure)} · B {format_validity_flag(baseline)}"
  
  
def format_numeric_or_text(value: Any, precision: int, default: str = "--") -> str:  
    if value in (None, "", "--"):  
        return default  
    try:  
        number = float(value)  
    except (TypeError, ValueError):  
        return str(value)  
    if not math.isfinite(number):  
        return default  
    return f"{number:.{precision}f}"  
  
  
def format_if_number(value: Any, precision: int = 3) -> Any:  
    if isinstance(value, (int, float)) and not isinstance(value, bool):  
        number = float(value)  
        if math.isfinite(number):  
            return f"{number:.{precision}f}"  
    return value  
  
  
def set_text_if_changed(label: QLabel, text: str) -> None:  
    if label.text() != text:  
        label.setText(text)  
  
  
def set_plaintext_if_changed(edit: QPlainTextEdit, text: str) -> None:  
    if edit.toPlainText() != text:  
        edit.setPlainText(text)  
  
  
def check_dependencies() -> None:  
    if not COMPOSE_COMMAND:  
        raise RuntimeError("LSIE_COMPOSE_COMMAND is empty.")  
    if shutil.which("ffmpeg") is None:  
        raise RuntimeError("ffmpeg is not installed on the host.")  
    if shutil.which(COMPOSE_COMMAND[0]) is None:  
        raise RuntimeError(f"Cannot find compose executable: {COMPOSE_COMMAND[0]!r}")  
  
  
def stop_process(proc: subprocess.Popen[bytes] | None) -> None:  
    if proc is None:  
        return  
    if proc.poll() is None:  
        proc.terminate()  
        try:  
            proc.wait(timeout=2)  
        except subprocess.TimeoutExpired:  
            proc.kill()  
            try:  
                proc.wait(timeout=1)  
            except subprocess.TimeoutExpired:  
                pass  
  
  
def read_exact(stream: Any, size: int) -> bytes:  
    chunks: list[bytes] = []  
    remaining = size  
    while remaining > 0:  
        chunk = stream.read(remaining)  
        if not chunk:  
            break  
        chunks.append(chunk)  
        remaining -= len(chunk)  
    return b"".join(chunks)  
  
  
def ndarray_to_qimage(image_bgr: np.ndarray) -> QImage:  
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  
    image_rgb = np.ascontiguousarray(image_rgb)  
    h, w, ch = image_rgb.shape  
    bytes_per_line = ch * w  
    return QImage(  
        image_rgb.data,  
        w,  
        h,  
        bytes_per_line,  
        QImage.Format.Format_RGB888,  
    ).copy()  
  
  
class LiveAU12Normalizer:  
    """  
    Self-contained live AU12 normalizer.  
    Computes IOD and mouth span ratio, calibrates neutral baseline, then outputs  
    bounded AU12 intensity.  
    """  
  
    def __init__(self, alpha: float = 6.0) -> None:  
        self.alpha = alpha  
        self.calibration_buffer: list[float] = []  
        self.b_neutral: float | None = None  
        self.is_calibrated = False  
        self.calibration_frames = 45  
  
    def reset(self) -> None:  
        self.calibration_buffer.clear()  
        self.b_neutral = None  
        self.is_calibrated = False  
  
    def process(self, landmarks: Any, w: int, h: int) -> float:  
        def pt(idx: int) -> tuple[float, float, float]:  
            lm = landmarks.landmark[idx]  
            return lm.x * w, lm.y * h, lm.z * w  
  
        left_eye_outer = pt(33)  
        left_eye_inner = pt(133)  
        right_eye_inner = pt(362)  
        right_eye_outer = pt(263)  
        left_lip = pt(61)  
        right_lip = pt(291)  
  
        left_eye_center = (  
            (left_eye_outer[0] + left_eye_inner[0]) * 0.5,  
            (left_eye_outer[1] + left_eye_inner[1]) * 0.5,  
            (left_eye_outer[2] + left_eye_inner[2]) * 0.5,  
        )  
        right_eye_center = (  
            (right_eye_inner[0] + right_eye_outer[0]) * 0.5,  
            (right_eye_inner[1] + right_eye_outer[1]) * 0.5,  
            (right_eye_inner[2] + right_eye_outer[2]) * 0.5,  
        )  
  
        dx = right_eye_center[0] - left_eye_center[0]  
        dy = right_eye_center[1] - left_eye_center[1]  
        dz = right_eye_center[2] - left_eye_center[2]  
        iod = math.sqrt(dx * dx + dy * dy + dz * dz)  
        if iod < 1e-6:  
            return 0.0  
  
        mx = right_lip[0] - left_lip[0]  
        my = right_lip[1] - left_lip[1]  
        mz = right_lip[2] - left_lip[2]  
        d_mouth = math.sqrt(mx * mx + my * my + mz * mz)  
        ratio = d_mouth / iod  
  
        if not self.is_calibrated:  
            self.calibration_buffer.append(ratio)  
            if len(self.calibration_buffer) >= self.calibration_frames:  
                self.b_neutral = float(sum(self.calibration_buffer) / len(self.calibration_buffer))  
                self.is_calibrated = True  
            return 0.0  
  
        if self.b_neutral is None:  
            return 0.0  
  
        deviation = max(0.0, ratio - self.b_neutral)  
        return float(math.tanh(self.alpha * deviation))  
  
  
def apply_overlay(  
    image: np.ndarray,  
    face_landmarks: Any | None,  
    mp_drawing: Any,  
    mp_face_mesh: Any,  
    overlay_mode: str,  
    au12_normalizer: LiveAU12Normalizer,  
    mesh_connection_spec: Any,  
) -> tuple[np.ndarray, dict[str, Any]]:  
    metrics: dict[str, Any] = {  
        "left_lip": "--",  
        "right_lip": "--",  
        "mouth_span": "--",  
        "face": "Searching",  
        "overlay": mode_label(overlay_mode),  
        "au12_val": 0.0,  
        "au12_status": "Waiting for face...",  
    }  
  
    left_xy: tuple[int, int] | None = None  
    right_xy: tuple[int, int] | None = None  
  
    if face_landmarks is not None:  
        h, w, _ = image.shape  
  
        au12_val = au12_normalizer.process(face_landmarks, w, h)  
        metrics["au12_val"] = au12_val  
  
        if not au12_normalizer.is_calibrated:  
            metrics["au12_status"] = (  
                f"Calibrating... "  
                f"({len(au12_normalizer.calibration_buffer)}/"  
                f"{au12_normalizer.calibration_frames})"  
            )  
        else:  
            metrics["au12_status"] = f"Active | B_neutral: {au12_normalizer.b_neutral:.3f}"  
  
        left_lip = face_landmarks.landmark[61]  
        right_lip = face_landmarks.landmark[291]  
  
        lx, ly = int(left_lip.x * w), int(left_lip.y * h)  
        rx, ry = int(right_lip.x * w), int(right_lip.y * h)  
  
        left_xy = (lx, ly)  
        right_xy = (rx, ry)  
  
        metrics["left_lip"] = f"{lx}, {ly}"  
        metrics["right_lip"] = f"{rx}, {ry}"  
        metrics["mouth_span"] = str(int(round(math.hypot(rx - lx, ry - ly))))  
        metrics["face"] = "Detected"  
  
    if overlay_mode == OVERLAY_RAW:  
        return image, metrics  
  
    if overlay_mode == OVERLAY_FULL and face_landmarks is not None:  
        overlay = image.copy()  
        mp_drawing.draw_landmarks(  
            image=overlay,  
            landmark_list=face_landmarks,  
            connections=mp_face_mesh.FACEMESH_TESSELATION,  
            landmark_drawing_spec=None,  
            connection_drawing_spec=mesh_connection_spec,  
        )  
        cv2.addWeighted(overlay, 0.20, image, 0.80, 0.0, dst=image)  
  
    if face_landmarks is not None and left_xy is not None and right_xy is not None:  
        if overlay_mode == OVERLAY_LIPS:  
            cv2.line(image, left_xy, right_xy, LIP_OUTLINE_COLOR, 4, cv2.LINE_AA)  
            cv2.line(image, left_xy, right_xy, LIP_MARKER_COLOR, 2, cv2.LINE_AA)  
  
        cv2.circle(image, left_xy, 6, LIP_OUTLINE_COLOR, -1)  
        cv2.circle(image, left_xy, 4, LIP_MARKER_COLOR, -1)  
  
        cv2.circle(image, right_xy, 6, LIP_OUTLINE_COLOR, -1)  
        cv2.circle(image, right_xy, 4, LIP_MARKER_COLOR, -1)  
  
    return image, metrics  
  
  
def start_stream_processes() -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes]]:  
    container_cmd = [  
        *COMPOSE_COMMAND,  
        "exec",  
        "-T",  
        STREAM_SERVICE,  
        "sh",  
        "-c",  
        (  
            "while true; do "  
            f"adb exec-out screenrecord --output-format=h264 --time-limit 180 "  
            f"--size {FRAME_WIDTH}x{FRAME_HEIGHT} --bit-rate 2000000 -; "  
            "done"  
        ),  
    ]  
  
    ffmpeg_cmd = [  
        "ffmpeg",  
        "-hide_banner",  
        "-loglevel",  
        "error",  
        "-f",  
        "h264",  
        "-i",  
        "pipe:0",  
        "-an",  
        "-vf",  
        (  
            f"fps={OUTPUT_FPS},"  
            f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease,"  
            f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2"  
        ),  
        "-pix_fmt",  
        "bgr24",  
        "-f",  
        "rawvideo",  
        "pipe:1",  
    ]  
  
    source_proc = subprocess.Popen(  
        container_cmd,  
        stdout=subprocess.PIPE,  
        stderr=subprocess.DEVNULL,  
        bufsize=0,  
    )  
    if source_proc.stdout is None:  
        raise RuntimeError("Failed to open compose exec stdout.")  
  
    decoder_proc = subprocess.Popen(  
        ffmpeg_cmd,  
        stdin=source_proc.stdout,  
        stdout=subprocess.PIPE,  
        stderr=subprocess.DEVNULL,  
        bufsize=0,  
    )  
  
    source_proc.stdout.close()  
    if decoder_proc.stdout is None:  
        stop_process(decoder_proc)  
        stop_process(source_proc)  
        raise RuntimeError("Failed to open ffmpeg stdout.")  
  
    return source_proc, decoder_proc  
  
  
class StreamThread(QThread):
    """LIVE PATH — reads frames from the IPC Pipe and runs MediaPipe in-process.

    See the data-consumption contract at the top of this module: this thread
    owns path (1) and MUST NOT read from PostgreSQL or the API.
    """

    frame_ready = Signal(QImage)
    status_changed = Signal(str, str)
    metrics_changed = Signal(object)
    log_message = Signal(str)
  
    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._proc_lock = threading.Lock()
        self._overlay_lock = threading.Lock()

        self._source_proc: subprocess.Popen[bytes] | None = None
        self._decoder_proc: subprocess.Popen[bytes] | None = None

        self._reconnects = 0
        # Windowed FPS measurement: ring of recent frame timestamps (last ~2s)
        # avoids the per-frame 1/dt artifact where pipe-buffer bursts produce
        # 300 fps spikes and quiet reads produce <10 fps dips. Actual
        # throughput is clamped upstream by `ffmpeg -vf fps=30`.
        self._frame_times: deque[float] = deque(maxlen=120)
        # Windowed-FPS samples (one per emit tick) for stable min/avg/max.
        self._fps_samples: deque[float] = deque(maxlen=60)
        self._overlay_mode = OVERLAY_FULL
        self._au12_normalizer = LiveAU12Normalizer()
        # Single-slot latest-frame decoupling. FFmpeg decodes faster than
        # MediaPipe + Qt paint when the processor occasionally stalls past the
        # 33 ms budget; without this, the OS pipe buffer fills and the
        # processor catches up by painting a burst of now-stale frames,
        # producing visible slow-mo stutter. The reader thread drains stdout
        # as fast as it arrives and keeps only the freshest frame; the
        # processor loop always paints the current moment.
        self._frame_cond = threading.Condition()
        self._latest_frame: bytes | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._dropped_frames = 0
  
    def stop(self) -> None:  
        self._stop_event.set()  
        self._stop_processes()  
  
    def request_restart(self) -> None:  
        self._restart_event.set()  
        self._stop_processes()  
  
    def set_overlay_mode(self, mode: str) -> None:  
        with self._overlay_lock:  
            self._overlay_mode = mode  
  
    def get_overlay_mode(self) -> str:  
        with self._overlay_lock:  
            return self._overlay_mode  
  
    def _stop_processes(self) -> None:
        # Signal reader to exit before tearing down its stdout, then close
        # processes, then join. Order matters: closing stdout unblocks the
        # reader's blocked read so the join completes quickly.
        self._reader_stop.set()
        with self._proc_lock:
            stop_process(self._decoder_proc)
            stop_process(self._source_proc)
            self._decoder_proc = None
            self._source_proc = None
        reader = self._reader_thread
        if reader is not None and reader.is_alive():
            reader.join(timeout=2.0)
        self._reader_thread = None
        with self._frame_cond:
            self._latest_frame = None
            self._frame_cond.notify_all()

    def _connect_fresh_stream(self) -> None:
        if self._stop_event.is_set():
            return
        source_proc, decoder_proc = start_stream_processes()
        with self._proc_lock:
            self._source_proc = source_proc
            self._decoder_proc = decoder_proc

        self._reader_stop.clear()
        with self._frame_cond:
            self._latest_frame = None
        self._dropped_frames = 0
        stdout = decoder_proc.stdout
        if stdout is None:
            raise EOFError("Decoder process has no stdout")
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(stdout,),
            name="StreamThread-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self, stdout: Any) -> None:
        """Drain the decoder's stdout at line rate into a single-slot buffer.

        Dropping the previous unread frame when a new one arrives is the
        whole point: the processor always paints the current moment, even
        when it briefly falls behind the 30 fps budget.
        """
        while not self._reader_stop.is_set():
            try:
                frame = read_exact(stdout, FRAME_BYTES)
            except (ValueError, OSError):
                break
            if len(frame) != FRAME_BYTES:
                break
            with self._frame_cond:
                if self._latest_frame is not None:
                    self._dropped_frames += 1
                self._latest_frame = frame
                self._frame_cond.notify()

    def _take_frame(self, timeout: float) -> bytes | None:
        """Return the freshest buffered frame, or None on timeout/shutdown."""
        with self._frame_cond:
            if self._latest_frame is None:
                self._frame_cond.wait(timeout=timeout)
            frame = self._latest_frame
            self._latest_frame = None
            return frame
  
    def run(self) -> None:  
        try:  
            check_dependencies()  
        except Exception as exc:  
            self.status_changed.emit("ERROR", "bad")  
            self.log_message.emit(f"Dependency check failed: {exc}")  
            return  
  
        self.log_message.emit("Worker started")  
  
        mp_drawing = mp.solutions.drawing_utils  
        mp_face_mesh = mp.solutions.face_mesh  
        mesh_connection_spec = mp_drawing.DrawingSpec(  
            color=(255, 255, 255),  
            thickness=1,  
            circle_radius=0,  
        )  
  
        try:  
            with mp_face_mesh.FaceMesh(  
                max_num_faces=1,  
                refine_landmarks=True,  
                min_detection_confidence=0.5,  
                min_tracking_confidence=0.5,  
            ) as face_mesh:  
                while not self._stop_event.is_set():  
                    try:  
                        self.status_changed.emit("CONNECTING", "warn")  
                        self._connect_fresh_stream()  
                        if self._stop_event.is_set():  
                            break  
  
                        self._frame_times.clear()
                        self._fps_samples.clear()
                        last_metrics_emit_time = 0.0

                        self.status_changed.emit("LIVE", "ok")
                        self.log_message.emit("Live stream connected")

                        fps_window_seconds = 1.0
  
                        while not self._stop_event.is_set() and not self._restart_event.is_set():
                            raw_frame = self._take_frame(timeout=1.0)
                            if raw_frame is None:
                                reader = self._reader_thread
                                if reader is None or not reader.is_alive():
                                    raise EOFError("Stream stalled or ended")
                                # Reader is alive but nothing arrived inside
                                # the timeout — let the loop re-check the
                                # stop/restart events and wait again.
                                continue
                            if len(raw_frame) != FRAME_BYTES:
                                raise EOFError("Stream stalled or ended")
  
                            image = np.frombuffer(raw_frame, dtype=np.uint8).reshape(  
                                (FRAME_HEIGHT, FRAME_WIDTH, 3)  
                            ).copy()  
  
                            image.flags.writeable = False  
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
                            results = face_mesh.process(image_rgb)  
                            image.flags.writeable = True  
  
                            face_landmarks = (  
                                results.multi_face_landmarks[0]  
                                if results.multi_face_landmarks  
                                else None  
                            )  
  
                            current_mode = self.get_overlay_mode()  
                            image, overlay_metrics = apply_overlay(  
                                image=image,  
                                face_landmarks=face_landmarks,  
                                mp_drawing=mp_drawing,  
                                mp_face_mesh=mp_face_mesh,  
                                overlay_mode=current_mode,  
                                au12_normalizer=self._au12_normalizer,  
                                mesh_connection_spec=mesh_connection_spec,  
                            )  
  
                            now = time.perf_counter()
                            self._frame_times.append(now)
                            # Drop timestamps older than the measurement window.
                            while (
                                self._frame_times
                                and now - self._frame_times[0] > fps_window_seconds
                            ):
                                self._frame_times.popleft()

                            self.frame_ready.emit(ndarray_to_qimage(image))

                            if now - last_metrics_emit_time >= METRICS_UPDATE_INTERVAL:
                                last_metrics_emit_time = now

                                if len(self._frame_times) >= 2:
                                    window_elapsed = (
                                        self._frame_times[-1] - self._frame_times[0]
                                    )
                                    current_fps = (
                                        (len(self._frame_times) - 1) / window_elapsed
                                        if window_elapsed > 0
                                        else 0.0
                                    )
                                else:
                                    current_fps = 0.0

                                self._fps_samples.append(current_fps)
                                fps_min = min(self._fps_samples)
                                fps_max = max(self._fps_samples)
                                fps_avg = sum(self._fps_samples) / len(self._fps_samples)

                                self.metrics_changed.emit(
                                    {
                                        "stream": "LIVE",
                                        "face": overlay_metrics["face"],
                                        "fps": f"{current_fps:.1f}",
                                        "fps_min": f"{fps_min:.1f}",
                                        "fps_max": f"{fps_max:.1f}",
                                        "fps_avg": f"{fps_avg:.1f}",
                                        "reconnects": str(self._reconnects),
                                        "overlay": overlay_metrics["overlay"],
                                        "au12_val": overlay_metrics["au12_val"],
                                        "au12_status": overlay_metrics["au12_status"],
                                        "left_lip": overlay_metrics["left_lip"],
                                        "right_lip": overlay_metrics["right_lip"],
                                        "mouth_span": overlay_metrics["mouth_span"],
                                    }
                                )
  
                        self._stop_processes()  
  
                        if self._restart_event.is_set() and not self._stop_event.is_set():  
                            self._restart_event.clear()  
                            self.status_changed.emit("RESTARTING", "warn")  
                            self.log_message.emit("Manual restart requested")  
                            if self._stop_event.wait(0.3):  
                                break  
  
                    except EOFError as exc:  
                        self._stop_processes()  
                        if self._stop_event.is_set():  
                            break  
                        if self._restart_event.is_set():  
                            self._restart_event.clear()  
                            self.status_changed.emit("RESTARTING", "warn")  
                            self.log_message.emit("Manual restart requested")  
                            if self._stop_event.wait(0.3):  
                                break  
                            continue  
  
                        self._reconnects += 1  
                        self.status_changed.emit("RECONNECTING", "warn")  
                        self.log_message.emit(str(exc))  
                        if self._stop_event.wait(1.0):  
                            break  
  
                    except subprocess.CalledProcessError as exc:  
                        self._stop_processes()  
                        if self._stop_event.is_set():  
                            break  
  
                        self._reconnects += 1  
                        self.status_changed.emit("ERROR", "bad")  
                        self.log_message.emit(f"Compose/stream command failed: {exc}")  
                        if self._stop_event.wait(1.5):  
                            break  
  
                    except Exception as exc:  
                        self._stop_processes()  
                        if self._stop_event.is_set():  
                            break  
  
                        self._reconnects += 1  
                        self.status_changed.emit("ERROR", "bad")  
                        self.log_message.emit(f"Worker exception: {exc}")  
                        if self._stop_event.wait(1.5):  
                            break  
  
        except Exception as exc:  
            if not self._stop_event.is_set():  
                self.status_changed.emit("ERROR", "bad")  
                self.log_message.emit(f"Worker fatal exception: {exc}")  
  
        finally:  
            self._stop_processes()  
            self.status_changed.emit("STOPPED", "neutral")  
            self.log_message.emit("Worker stopped")  
  
  
class HardwareTelemetryThread(QThread):
    """HARDWARE PATH — nvidia-smi and adb out-of-band.

    See the data-consumption contract at the top of this module: host/device
    telemetry never transits the API or the Persistent Store.
    """

    hw_stats_ready = Signal(dict)
    log_message = Signal(str)
  
    def __init__(self) -> None:  
        super().__init__()  
        self._stop_event = threading.Event()  
        self._host_gpu_name = ""  
        self._device_profile: dict[str, str] | None = None  
        self._latest_device_identity = ""  
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None  
        self._device_profile_retry_after = 0.0  
        self._last_payload: dict[str, str] | None = None  
  
        if psutil is not None:  
            try:  
                psutil.cpu_percent(interval=None)  
            except Exception:  
                pass  
  
    def stop(self) -> None:  
        self._stop_event.set()  
  
    def _adb_shell(self, script: str, timeout: float = 5.0) -> str:  
        cmd = [  
            *COMPOSE_COMMAND,  
            "exec",  
            "-T",  
            STREAM_SERVICE,  
            "adb",  
            "shell",  
            "sh",  
            "-c",  
            script,  
        ]  
        res = subprocess.run(  
            cmd,  
            capture_output=True,  
            text=True,  
            timeout=timeout,  
            check=False,  
        )  
        if res.returncode != 0:  
            raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "adb shell failed")  
        return res.stdout  
  
    def _get_host_stats(self) -> dict[str, str]:  
        stats = {  
            "cpu": "--%",  
            "gpu": "--%",  
            "gpu_hint": "NVIDIA SMI unavailable",  
        }  
  
        if psutil is not None:  
            try:  
                stats["cpu"] = f"{psutil.cpu_percent(interval=None):.1f}%"  
            except Exception:  
                pass  
        else:  
            stats["cpu"] = "(psutil missing)"  
  
        if not self._has_nvidia_smi:  
            return stats  
  
        try:  
            if not self._host_gpu_name:  
                res_name = subprocess.run(  
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  
                    capture_output=True,  
                    text=True,  
                    timeout=2,  
                    check=False,  
                )  
                if res_name.returncode == 0:  
                    first_line = res_name.stdout.strip().splitlines()  
                    if first_line:  
                        self._host_gpu_name = first_line[0].strip()  
  
            res = subprocess.run(  
                [  
                    "nvidia-smi",  
                    "--query-gpu=utilization.gpu,memory.used,memory.total",  
                    "--format=csv,noheader,nounits",  
                ],  
                capture_output=True,  
                text=True,  
                timeout=2,  
                check=False,  
            )  
            if res.returncode == 0:  
                line = res.stdout.strip().splitlines()  
                if line:  
                    parts = [p.strip() for p in line[0].split(",")]  
                    if len(parts) >= 3:  
                        stats["gpu"] = f"{parts[0]}%"  
                        stats["gpu_hint"] = (  
                            f"{self._host_gpu_name or 'NVIDIA GPU'} · {parts[1]}/{parts[2]} MB"  
                        )  
        except Exception:  
            pass  
  
        return stats  
  
    def _detect_device_profile(self) -> dict[str, str]:  
        script = r"""  
manufacturer=$(getprop ro.product.manufacturer 2>/dev/null)  
model=$(getprop ro.product.model 2>/dev/null)  
android=$(getprop ro.build.version.release 2>/dev/null)  
soc=$(getprop ro.soc.model 2>/dev/null)  
board=$(getprop ro.board.platform 2>/dev/null)  
hardware=$(getprop ro.hardware 2>/dev/null)  
gles=$(dumpsys SurfaceFlinger 2>/dev/null | sed -n '/GLES:/ { s/^GLES: //; p; q; }')  
echo "${manufacturer}|${model}|${android}|${soc}|${board}|${hardware}|${gles}"  
"""  
        out = self._adb_shell(script, timeout=4).strip()  
        parts = out.split("|")  
        parts += [""] * max(0, 7 - len(parts))  
        manufacturer, model, android, soc, board, hardware, gles = parts[:7]  
  
        gpu_name = "--"  
        if gles:  
            gparts = [p.strip() for p in gles.split(",", 2)]  
            gpu_name = gparts[1] if len(gparts) >= 2 else gles.strip()  
  
        identity_parts = [  
            f"{manufacturer} {model}".strip(),  
            f"Android {android}" if android else "",  
            soc or board or hardware,  
        ]  
        identity = " · ".join(p for p in identity_parts if p) or "Android device"  
  
        return {  
            "identity": identity,  
            "gpu_name": gpu_name,  
        }  
  
    def _ensure_device_profile(self) -> None:  
        if self._device_profile is not None:  
            return  
  
        now = time.monotonic()  
        if now < self._device_profile_retry_after:  
            return  
  
        try:  
            self._device_profile = self._detect_device_profile()  
            self._latest_device_identity = self._device_profile["identity"]  
            self.log_message.emit(  
                f"Detected device: {self._device_profile['identity']} · "  
                f"GPU: {self._device_profile['gpu_name']}"  
            )  
        except Exception:  
            self._device_profile_retry_after = now + 10.0  
  
    def _get_device_stats(self) -> dict[str, str]:  
        self._ensure_device_profile()  
        profile = self._device_profile or {  
            "identity": "Android device",  
            "gpu_name": "--",  
        }  
  
        stats = {  
            "cpu": "--%",  
            "cpu_hint": "Waiting for ADB...",  
            "fg_app": "Unknown",  
            "fg_app_hint": "",  
        }  
  
        script = r"""  
btemp=$(dumpsys battery 2>/dev/null | awk '/temperature:/ {print $2; exit}')  
  
read cpu u n s i io irq sirq st _ _ < /proc/stat  
tot1=$((u+n+s+i+io+irq+sirq+st))  
idle1=$((i+io))  
  
sr1=0  
for p in $(pidof screenrecord 2>/dev/null); do  
  v=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null)  
  [ -n "$v" ] && sr1=$((sr1 + v))  
done  
  
sleep 0.5  
  
read cpu u n s i io irq sirq st _ _ < /proc/stat  
tot2=$((u+n+s+i+io+irq+sirq+st))  
idle2=$((i+io))  
  
sr2=0  
for p in $(pidof screenrecord 2>/dev/null); do  
  v=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null)  
  [ -n "$v" ] && sr2=$((sr2 + v))  
done  
  
dt=$((tot2 - tot1))  
didle=$((idle2 - idle1))  
dsr=$((sr2 - sr1))  
[ "$dsr" -lt 0 ] && dsr=0  
  
if [ "$dt" -gt 0 ]; then  
  cpu_used=$((100 * (dt - didle) / dt))  
  sr_cpu=$((100 * dsr / dt))  
else  
  cpu_used="--"  
  sr_cpu="--"  
fi  
  
focus=$(dumpsys activity activities 2>/dev/null | awk '/ResumedActivity:/ {print $4}' | head -n 1 | cut -d/ -f1)  
if [ -z "$focus" ]; then  
  focus=$(dumpsys window 2>/dev/null | awk '/mCurrentFocus=Window/ {print $3}' | head -n 1 | cut -d/ -f1)  
fi  
  
echo "${btemp}|${cpu_used}|${sr_cpu}|${focus}"  
"""  
        try:  
            out = self._adb_shell(script, timeout=5).strip()  
            parts = out.split("|")  
            if len(parts) >= 3:  
                btemp_raw, cpu_used, sr_cpu = parts[0], parts[1], parts[2]  
  
                stats["fg_app"] = "Home Screen"  
                if len(parts) >= 4:  
                    pkg = parts[3].strip().replace("}", "")  
                    if pkg and pkg not in ("null", "--"):  
                        app_map = {  
                            "com.zhiliaoapp.musically": "TikTok",  
                            "com.ss.android.ugc.trill": "TikTok",  
                            "com.google.android.youtube": "YouTube",  
                            "com.instagram.android": "Instagram",  
                            "com.facebook.katana": "Facebook",  
                            "com.google.android.apps.nexuslauncher": "Home Screen",  
                            "com.sec.android.app.launcher": "Home Screen",  
                            "com.android.launcher": "Home Screen",  
                            "com.android.chrome": "Chrome",  
                            "com.google.android.gm": "Gmail",  
                        }  
                        stats["fg_app"] = app_map.get(pkg, pkg)  
                        if stats["fg_app"] == pkg and pkg not in ("None", "Home Screen"):  
                            stats["fg_app_hint"] = "Package Name"  
                        elif stats["fg_app"] != pkg:  
                            stats["fg_app_hint"] = pkg  
  
                if cpu_used != "--":  
                    stats["cpu"] = f"{cpu_used}%"  
  
                batt_temp = "--"  
                if btemp_raw.isdigit():  
                    batt_temp = f"{int(btemp_raw) / 10.0:.1f}°C"  
  
                self._latest_device_identity = profile["identity"]  
  
                live_parts: list[str] = []  
                if sr_cpu != "--":  
                    live_parts.append(f"capture: {sr_cpu}%")  
                if batt_temp != "--":  
                    live_parts.append(f"batt: {batt_temp}")  
  
                stats["cpu_hint"] = " · ".join(live_parts) if live_parts else "Polling..."  
        except Exception:  
            stats["cpu_hint"] = "ADB unreachable"  
  
        return stats  
  
    def get_latest_device_identity(self) -> str:  
        return self._latest_device_identity  
  
    def run(self) -> None:  
        self.log_message.emit("Hardware telemetry thread started")  
  
        while not self._stop_event.is_set():  
            host_stats = self._get_host_stats()  
            dev_stats = self._get_device_stats()  
  
            payload = {  
                "host_cpu": host_stats["cpu"],  
                "host_gpu": host_stats["gpu"],  
                "host_gpu_hint": host_stats["gpu_hint"],  
                "dev_cpu": dev_stats["cpu"],  
                "dev_cpu_hint": dev_stats["cpu_hint"],  
                "fg_app": dev_stats.get("fg_app", "Unknown"),  
                "fg_app_hint": dev_stats.get("fg_app_hint", ""),  
            }  
  
            if payload != self._last_payload:  
                self._last_payload = payload  
                self.hw_stats_ready.emit(payload)  
  
            self._stop_event.wait(2.0)  
  
        self.log_message.emit("Hardware telemetry thread stopped")  
  
  
class AnalyticsThread(QThread):
    """ANALYTICS PATH — polls PostgreSQL via OPERATOR_SNAPSHOT_SQL.

    See the data-consumption contract at the top of this module. Every panel
    that shows persisted session state (right-hand side on the Live tab, and
    the entire Analytics tab) is fed from a single snapshot emitted here. Do
    not add HTTP calls to this thread — GUI writes go through the API, but
    GUI reads stay on psql.
    """

    snapshot_ready = Signal(object)
    log_message = Signal(str)
  
    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()
        self._last_snapshot_fingerprint = ""
        self._last_error = ""
        self._physio_available = True

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self.log_message.emit("DB polling thread started (bypassing HTTP)")
        while not self._stop_event.is_set():
            try:
                snapshot = run_psql_json(OPERATOR_SNAPSHOT_SQL)
                if self._physio_available:
                    try:
                        physio = run_psql_json(OPERATOR_PHYSIO_SQL)
                    except RuntimeError as physio_exc:
                        if "does not exist" in str(physio_exc):
                            self._physio_available = False
                            self.log_message.emit(
                                "Physiology tables missing — run migrations in "
                                "data/sql/03-physiology.sql to enable the "
                                "Analytics-tab physiology panels"
                            )
                        else:
                            raise
                    else:
                        if physio:
                            snapshot.update(physio)

                if snapshot:
                    fingerprint = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
                    if fingerprint != self._last_snapshot_fingerprint:
                        self._last_snapshot_fingerprint = fingerprint
                        self.snapshot_ready.emit(snapshot)
                self._last_error = ""
            except Exception as exc:
                msg = f"DB poll error: {exc}"
                if msg != self._last_error:
                    self._last_error = msg
                    self.log_message.emit(msg)

            self._stop_event.wait(ANALYTICS_POLL_SECONDS)

        self.log_message.emit("DB polling thread stopped")
  
  
class CustomTitleBar(QWidget):  
    def __init__(self, parent: QMainWindow):  
        super().__init__(parent)  
        self.parent_window = parent  
        self.setObjectName("TitleBar")  
        self.setFixedHeight(36)  
  
        layout = QHBoxLayout(self)  
        layout.setContentsMargins(16, 0, 0, 0)  
        layout.setSpacing(0)  
  
        title_label = QLabel(APP_NAME)  
        title_label.setStyleSheet(  
            "color: #64748B; font-weight: 600; font-size: 12px; letter-spacing: 0.5px;"  
        )  
  
        layout.addWidget(title_label)  
        layout.addStretch(1)  
  
        btn_min = QPushButton("—")  
        btn_min.setObjectName("SysButton")  
        btn_min.setFixedSize(46, 36)  
        btn_min.clicked.connect(self.parent_window.showMinimized)  
  
        btn_max = QPushButton("☐")  
        btn_max.setObjectName("SysButton")  
        btn_max.setFixedSize(46, 36)  
        btn_max.clicked.connect(self._toggle_maximize)  
  
        btn_close = QPushButton("✕")  
        btn_close.setObjectName("SysCloseButton")  
        btn_close.setFixedSize(46, 36)  
        btn_close.clicked.connect(self.parent_window.close)  
  
        layout.addWidget(btn_min)  
        layout.addWidget(btn_max)  
        layout.addWidget(btn_close)  
  
    def _toggle_maximize(self) -> None:  
        if self.parent_window.isMaximized():  
            self.parent_window.showNormal()  
        else:  
            self.parent_window.showMaximized()  
  
    def mousePressEvent(self, event) -> None:  
        if event.button() == Qt.MouseButton.LeftButton:  
            win_handle = self.window().windowHandle()  
            if win_handle:  
                win_handle.startSystemMove()  
            event.accept()  
  
    def mouseDoubleClickEvent(self, event) -> None:  
        if event.button() == Qt.MouseButton.LeftButton:  
            self._toggle_maximize()  
            event.accept()  
  
  
class StatusBadge(QLabel):  
    def __init__(self) -> None:  
        super().__init__()  
        self._current_state = ("", "")  
        self.setFixedSize(132, 32)  
        self.setAlignment(Qt.AlignCenter)  
        self.set_status("BOOTING", "neutral")  
  
    def set_status(self, text: str, level: str) -> None:  
        state = (text, level)  
        if state == self._current_state:  
            return  
        self._current_state = state  
  
        colors = {  
            "ok": ("rgba(52, 211, 153, 0.15)", "#34D399"),  
            "warn": ("rgba(251, 191, 36, 0.15)", "#FBBF24"),  
            "bad": ("rgba(248, 113, 113, 0.15)", "#F87171"),  
            "neutral": ("rgba(148, 163, 184, 0.15)", "#94A3B8"),  
        }  
        bg, fg = colors.get(level, colors["neutral"])  
        self.setStyleSheet(  
            f"""  
            QLabel {{  
                background-color: {bg};  
                color: {fg};  
                border: 1px solid rgba(255, 255, 255, 0.05);  
                border-radius: 16px;  
                font-weight: 700;  
                font-size: 11px;  
                letter-spacing: 1px;  
            }}  
            """  
        )  
        self.setText(f"● {text}")  
  
  
class MetricTile(QFrame):  
    def __init__(self, title: str, value: str = "--", hint: str = "") -> None:  
        super().__init__()  
        self.setObjectName("Card")  
        self.setMinimumHeight(86)  
  
        layout = QVBoxLayout(self)  
        layout.setContentsMargins(16, 16, 16, 16)  
        layout.setSpacing(4)  
  
        self.title_label = QLabel(title)  
        self.title_label.setObjectName("CardTitle")  
  
        self.value_label = QLabel(value)  
        self.value_label.setObjectName("CardValue")  
        self.value_label.setWordWrap(True)  
  
        self.hint_label = QLabel(hint)  
        self.hint_label.setObjectName("CardHint")  
        self.hint_label.setWordWrap(True)  
  
        layout.addWidget(self.title_label)  
        layout.addWidget(self.value_label)  
        layout.addWidget(self.hint_label)  
  
        self.hint_label.setVisible(bool(hint))  
  
    def set_value(self, value: Any) -> None:  
        set_text_if_changed(self.value_label, str(value))  
  
    def set_hint(self, hint: Any) -> None:  
        hint_text = str(hint)  
        set_text_if_changed(self.hint_label, hint_text)  
        self.hint_label.setVisible(bool(hint_text))  
  
  
class SecondaryTile(QFrame):  
    def __init__(self, title: str, value: str = "--", hint: str = "") -> None:  
        super().__init__()  
        self.setObjectName("Card")  
        self.setMinimumHeight(86)  
  
        layout = QVBoxLayout(self)  
        layout.setContentsMargins(16, 16, 16, 16)  
        layout.setSpacing(4)  
  
        self.title_label = QLabel(title)  
        self.title_label.setObjectName("CardTitle")  
  
        self.value_label = QLabel(value)  
        self.value_label.setObjectName("SecondaryValue")  
        self.value_label.setWordWrap(True)  
  
        self.hint_label = QLabel(hint)  
        self.hint_label.setObjectName("CardHint")  
        self.hint_label.setWordWrap(True)  
  
        layout.addWidget(self.title_label)  
        layout.addWidget(self.value_label)  
        layout.addWidget(self.hint_label)  
  
        self.hint_label.setVisible(bool(hint))  
  
    def set_value(self, value: Any) -> None:  
        set_text_if_changed(self.value_label, str(value))  
  
    def set_hint(self, hint: Any) -> None:  
        hint_text = str(hint)  
        set_text_if_changed(self.hint_label, hint_text)  
        self.hint_label.setVisible(bool(hint_text))  
  
  
class InlineStat(QFrame):  
    def __init__(self, title: str, value: str = "--", hint: str = "", compact: bool = False) -> None:  
        super().__init__()  
        self.setObjectName("InlineStat")  
  
        layout = QVBoxLayout(self)  
        layout.setContentsMargins(12, 10, 12, 10)  
        layout.setSpacing(2)  
  
        self.title_label = QLabel(title)  
        self.title_label.setObjectName("InlineTitle")  
  
        self.value_label = QLabel(value)  
        self.value_label.setObjectName("InlineValueCompact" if compact else "InlineValue")  
        self.value_label.setWordWrap(True)  
  
        self.hint_label = QLabel(hint)  
        self.hint_label.setObjectName("CardHint")  
        self.hint_label.setWordWrap(True)  
  
        layout.addWidget(self.title_label)  
        layout.addWidget(self.value_label)  
        layout.addWidget(self.hint_label)  
  
        self.hint_label.setVisible(bool(hint))  
  
    def set_value(self, value: Any) -> None:  
        set_text_if_changed(self.value_label, str(value))  
  
    def set_hint(self, hint: Any) -> None:  
        hint_text = str(hint)  
        set_text_if_changed(self.hint_label, hint_text)  
        self.hint_label.setVisible(bool(hint_text))  
  
  
class PanelCard(QFrame):  
    def __init__(self, title: str, header_widget: QWidget | None = None) -> None:  
        super().__init__()  
        self.setObjectName("Card")  
  
        outer_layout = QVBoxLayout(self)  
        outer_layout.setContentsMargins(20, 16, 20, 16)  
        outer_layout.setSpacing(12)  
  
        header = QHBoxLayout()  
        header.setContentsMargins(0, 0, 0, 0)  
        header.setSpacing(8)  
  
        self.title_label = QLabel(title)  
        self.title_label.setObjectName("PanelTitle")  
        header.addWidget(self.title_label)  
        header.addStretch(1)  
  
        if header_widget is not None:  
            header.addWidget(header_widget)  
  
        self.body_layout = QVBoxLayout()  
        self.body_layout.setContentsMargins(0, 0, 0, 0)  
        self.body_layout.setSpacing(12)  
  
        outer_layout.addLayout(header)  
        outer_layout.addLayout(self.body_layout)  
  
  
class IntensityBarWidget(QWidget):  
    def __init__(self) -> None:  
        super().__init__()  
        self._value = 0.0  
        self.setMinimumHeight(24)  
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  
  
    def set_value(self, val: float) -> None:  
        new_value = max(0.0, min(1.0, float(val)))  
        if abs(new_value - self._value) < 1e-4:  
            return  
        self._value = new_value  
        self.update()  
  
    def paintEvent(self, event) -> None:  
        painter = QPainter(self)  
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  
  
        r = self.rect().adjusted(0, 6, 0, -6)  
  
        painter.setPen(Qt.NoPen)  
        painter.setBrush(QColor(255, 255, 255, 15))  
        painter.drawRoundedRect(r, 4, 4)  
  
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))  
        for pct in (0.25, 0.5, 0.75):  
            x = r.left() + r.width() * pct  
            painter.drawLine(int(x), r.top(), int(x), r.bottom())  
  
        if self._value > 0.001:  
            fill_w = r.width() * self._value  
            fill_r = QRectF(r.left(), r.top(), fill_w, r.height())  
  
            gradient = QLinearGradient(r.topLeft(), r.topRight())  
            gradient.setColorAt(0.0, QColor("#3B82F6"))  
            gradient.setColorAt(0.5, QColor("#8B5CF6"))  
            gradient.setColorAt(1.0, QColor("#F43F5E"))  
  
            painter.setBrush(gradient)  
            painter.setPen(Qt.NoPen)  
            painter.drawRoundedRect(fill_r, 4, 4)  
  
  
class AU12Tile(QFrame):  
    def __init__(self, title: str, value: str = "--", hint: str = "") -> None:  
        super().__init__()  
        self.setObjectName("Card")  
        self.setMinimumHeight(110)  
  
        layout = QVBoxLayout(self)  
        layout.setContentsMargins(16, 16, 16, 16)  
        layout.setSpacing(4)  
  
        self.title_label = QLabel(title)  
        self.title_label.setObjectName("CardTitle")  
  
        self.value_label = QLabel(value)  
        self.value_label.setObjectName("CardValue")  
  
        self.hint_label = QLabel(hint)  
        self.hint_label.setObjectName("CardHint")  
        self.hint_label.setWordWrap(True)  
  
        self.bar = IntensityBarWidget()  
  
        layout.addWidget(self.title_label)  
        layout.addWidget(self.value_label)  
        layout.addWidget(self.hint_label)  
        layout.addWidget(self.bar)  
  
        self.hint_label.setVisible(bool(hint))  
  
    def set_value(self, str_val: str, numeric_val: float) -> None:  
        set_text_if_changed(self.value_label, str(str_val))  
        self.bar.set_value(numeric_val)  
  
    def set_hint(self, hint: Any) -> None:  
        hint_text = str(hint)  
        set_text_if_changed(self.hint_label, hint_text)  
        self.hint_label.setVisible(bool(hint_text))  
  
  
class TrackingPointsTile(QFrame):  
    def __init__(self) -> None:  
        super().__init__()  
        self.setObjectName("Card")  
  
        layout = QVBoxLayout(self)  
        layout.setContentsMargins(16, 16, 16, 16)  
        layout.setSpacing(4)  
  
        self.title_label = QLabel("Tracking Points")  
        self.title_label.setObjectName("CardTitle")  
  
        self.left_lip_label = QLabel("Left lip: --")  
        self.left_lip_label.setObjectName("InfoRow")  
  
        self.right_lip_label = QLabel("Right lip: --")  
        self.right_lip_label.setObjectName("InfoRow")  
  
        self.mouth_span_label = QLabel("Mouth span: --")  
        self.mouth_span_label.setObjectName("InfoRow")  
  
        layout.addWidget(self.title_label)  
        layout.addWidget(self.left_lip_label)  
        layout.addWidget(self.right_lip_label)  
        layout.addWidget(self.mouth_span_label)  
  
    def set_values(self, left_lip: Any, right_lip: Any, mouth_span: Any) -> None:  
        left_text = f"Left lip: {left_lip}"  
        right_text = f"Right lip: {right_lip}"  
        span_value = str(mouth_span)  
        span_text = f"Mouth span: {span_value} px" if span_value not in ("", "--") else "Mouth span: --"  
  
        set_text_if_changed(self.left_lip_label, left_text)  
        set_text_if_changed(self.right_lip_label, right_text)  
        set_text_if_changed(self.mouth_span_label, span_text)  
  
  
class SparklineWidget(QWidget):  
    def __init__(self, color_hex: str = "#3B82F6") -> None:  
        super().__init__()  
        self._values: list[float] = []  
        self.color_hex = color_hex  
        self.setMinimumHeight(60)  
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  
  
    def set_values(self, values: list[float]) -> None:  
        cleaned: list[float] = []  
        for value in values[-90:]:  
            try:  
                number = float(value)  
            except (TypeError, ValueError):  
                continue  
            if math.isfinite(number):  
                cleaned.append(number)  
        self._values = cleaned  
        self.update()  
  
    def paintEvent(self, event) -> None:  
        painter = QPainter(self)  
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  
  
        r = self.rect()  
  
        if len(self._values) < 2:  
            painter.setPen(QColor("#64748B"))  
            painter.drawText(r, Qt.AlignCenter, "Waiting for data…")  
            return  
  
        min_v = min(self._values)  
        max_v = max(self._values)  
        if abs(max_v - min_v) < 0.01:  
            max_v += 1.0  
            min_v -= 1.0  
  
        painter.setPen(QPen(QColor(255, 255, 255, 10), 1))  
        for frac in (0.25, 0.5, 0.75):  
            y = r.top() + frac * r.height()  
            painter.drawLine(r.left(), int(y), r.right(), int(y))  
  
        points: list[tuple[float, float]] = []  
        count = len(self._values)  
        for i, value in enumerate(self._values):  
            x = r.left() + (i / max(count - 1, 1)) * r.width()  
            norm = (value - min_v) / (max_v - min_v)  
            y = r.bottom() - 10 - norm * (r.height() - 20)  
            points.append((x, y))  
  
        line_path = QPainterPath()  
        line_path.moveTo(points[0][0], points[0][1])  
        for x, y in points[1:]:  
            line_path.lineTo(x, y)  
  
        fill_path = QPainterPath(line_path)  
        fill_path.lineTo(points[-1][0], r.bottom())  
        fill_path.lineTo(points[0][0], r.bottom())  
        fill_path.closeSubpath()  
  
        base_color = QColor(self.color_hex)  
        grad_start = QColor(base_color.red(), base_color.green(), base_color.blue(), 60)  
        grad_end = QColor(base_color.red(), base_color.green(), base_color.blue(), 0)  
  
        gradient = QLinearGradient(0, r.top(), 0, r.bottom())  
        gradient.setColorAt(0.0, grad_start)  
        gradient.setColorAt(1.0, grad_end)  
  
        painter.setPen(Qt.NoPen)  
        painter.setBrush(gradient)  
        painter.drawPath(fill_path)  
  
        painter.setPen(QPen(base_color, 2))  
        painter.setBrush(Qt.NoBrush)  
        painter.drawPath(line_path)  
  
        last_x, last_y = points[-1]  
        painter.setPen(Qt.NoPen)  
        painter.setBrush(base_color)  
        painter.drawEllipse(QRectF(last_x - 4, last_y - 4, 8, 8))  
  
        painter.setPen(QColor("#94A3B8"))  
        font = painter.font()  
        font.setPointSize(9)  
        painter.setFont(font)  
        painter.drawText(  
            r.adjusted(0, 4, 0, 0),  
            Qt.AlignTop | Qt.AlignLeft,  
            f"min {min(self._values):.1f}",  
        )  
        painter.drawText(  
            r.adjusted(0, 4, 0, 0),  
            Qt.AlignTop | Qt.AlignRight,  
            f"max {max(self._values):.1f}",  
        )  
  
  
class InlineSparkline(QFrame):  
    def __init__(  
        self,  
        title: str,  
        value: str = "--",  
        hint: str = "",  
        color_hex: str = "#3B82F6",  
    ) -> None:  
        super().__init__()  
        self.setObjectName("InlineStat")  
  
        layout = QVBoxLayout(self)  
        layout.setContentsMargins(12, 10, 12, 10)  
        layout.setSpacing(4)  
  
        self.title_label = QLabel(title)  
        self.title_label.setObjectName("InlineTitle")  
  
        self.value_label = QLabel(value)  
        self.value_label.setObjectName("InlineValue")  
  
        self.hint_label = QLabel(hint)  
        self.hint_label.setObjectName("CardHint")  
        self.hint_label.setVisible(bool(hint))  
  
        self.sparkline = SparklineWidget(color_hex)  
  
        layout.addWidget(self.title_label)  
        layout.addWidget(self.value_label)  
        layout.addWidget(self.hint_label)  
        layout.addWidget(self.sparkline)  
  
    def set_value(self, value: Any) -> None:  
        set_text_if_changed(self.value_label, str(value))  
  
    def set_hint(self, hint: Any) -> None:  
        hint_text = str(hint)  
        set_text_if_changed(self.hint_label, hint_text)  
        self.hint_label.setVisible(bool(hint_text))  
  
    def set_values(self, values: list[float]) -> None:  
        self.sparkline.set_values(values)  
  
  
class VideoPane(QFrame):  
    """  
    Lightweight video renderer.  
    Avoids extra pixmap scaling allocations on every paint.  
    """  
  
    def __init__(self) -> None:  
        super().__init__()  
        self._pixmap: QPixmap | None = None  
        self._placeholder = "Waiting for live frame…"  
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  
        self.setMinimumSize(320, 420)  
  
    def sizeHint(self) -> QSize:  
        return QSize(580, 800)  
  
    def set_frame(self, image: QImage) -> None:  
        self._pixmap = QPixmap.fromImage(image)  
        self.update()  
  
    def paintEvent(self, event) -> None:  
        painter = QPainter(self)  
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  
  
        avail = self.rect().adjusted(16, 16, -16, -16)  
  
        if self._pixmap is None:  
            painter.setPen(QColor("#64748B"))  
            font = painter.font()  
            font.setPointSize(15)  
            font.setBold(True)  
            painter.setFont(font)  
            painter.drawText(avail, Qt.AlignCenter, self._placeholder)  
            return  
  
        avail_w = max(50.0, float(avail.width()))  
        avail_h = max(50.0, float(avail.height()))  
  
        if avail_w / avail_h > FRAME_ASPECT:  
            phone_h = avail_h  
            phone_w = phone_h * FRAME_ASPECT  
        else:  
            phone_w = avail_w  
            phone_h = phone_w / FRAME_ASPECT  
  
        phone_rect = QRectF(  
            avail.center().x() - phone_w / 2.0,  
            avail.center().y() - phone_h / 2.0,  
            phone_w,  
            phone_h,  
        )  
  
        shell_rect = phone_rect.adjusted(-6, -6, 6, 6)  
        painter.setBrush(QColor("#020617"))  
        painter.setPen(QPen(QColor("#1E293B"), 1))  
        painter.drawRoundedRect(shell_rect, 16, 16)  
  
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)  
        painter.drawPixmap(phone_rect.toRect(), self._pixmap, self._pixmap.rect())  
  
  
class MainWindow(QMainWindow):  
    def __init__(self) -> None:  
        super().__init__()  
        self.setWindowTitle(APP_NAME)  
        self.setMinimumSize(1200, 800)  
        self.resize(1740, 1060)  
  
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)  
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  
  
        self.thread = StreamThread()  
        self.analytics_thread = AnalyticsThread()  
        self.hw_thread = HardwareTelemetryThread()  
  
        self._build_ui()  
        self._connect_signals()  
        self._set_overlay_mode(OVERLAY_FULL, log_message=False)  
  
        self.thread.start()  
        self.analytics_thread.start()  
        self.hw_thread.start()  
  
        self._append_log("UI ready")  
        self._append_log("Orchestrator MUST be running to populate analytics")  
        self._append_log(  
            "Shortcuts: 1 raw, 2 lip points, 3 face mesh, R restart UI stream, C clear log, Esc quit"  
        )  
  
    def _build_ui(self) -> None:  
        self.setStyleSheet(APP_STYLESHEET)  
  
        central = QWidget()  
        self.setCentralWidget(central)  
  
        self.root_layout = QVBoxLayout(central)  
        self.root_layout.setContentsMargins(24, 24, 24, 24)  
        self.root_layout.setSpacing(0)  
  
        self.main_frame = QFrame()  
        self.main_frame.setObjectName("MainFrame")  
        self.main_frame.setProperty("maximized", "false")  
  
        self.shadow = QGraphicsDropShadowEffect(self.main_frame)  
        self.shadow.setBlurRadius(40)  
        self.shadow.setColor(QColor(0, 0, 0, 180))  
        self.shadow.setOffset(0, 15)  
        self.shadow.setEnabled(True)  
        self.main_frame.setGraphicsEffect(self.shadow)  
  
        main_layout = QVBoxLayout(self.main_frame)  
        main_layout.setContentsMargins(0, 0, 0, 0)  
        main_layout.setSpacing(0)  
  
        self.title_bar = CustomTitleBar(self)  
        main_layout.addWidget(self.title_bar)  
  
        content_widget = QWidget()  
        content_layout = QVBoxLayout(content_widget)  
        content_layout.setContentsMargins(32, 24, 32, 32)  
        content_layout.setSpacing(24)  
  
        header = QHBoxLayout()  
        header.setSpacing(16)  
  
        title_block = QVBoxLayout()  
        title_block.setSpacing(4)  
  
        title = QLabel(APP_NAME)  
        title.setObjectName("Title")  
  
        subtitle = QLabel(  
            "Live MediaPipe overlay for stream_scrcpy · polished single-consumer debug studio"  
        )  
        subtitle.setObjectName("Subtitle")  
  
        title_block.addWidget(title)  
        title_block.addWidget(subtitle)  
  
        mode_container = QFrame()  
        mode_container.setObjectName("SegmentedControl")  
        mode_bar = QHBoxLayout(mode_container)  
        mode_bar.setContentsMargins(4, 4, 4, 4)  
        mode_bar.setSpacing(2)  
  
        self.mode_raw = QPushButton("Raw")  
        self.mode_lips = QPushButton("Lip Points")  
        self.mode_mesh = QPushButton("Face Mesh")  
  
        for btn in (self.mode_raw, self.mode_lips, self.mode_mesh):  
            btn.setCheckable(True)  
            btn.setProperty("modeButton", True)  
  
        self.mode_group = QButtonGroup(self)  
        self.mode_group.setExclusive(True)  
        self.mode_group.addButton(self.mode_raw)  
        self.mode_group.addButton(self.mode_lips)  
        self.mode_group.addButton(self.mode_mesh)  
        self.mode_mesh.setChecked(True)  
  
        mode_bar.addWidget(self.mode_raw)  
        mode_bar.addWidget(self.mode_lips)  
        mode_bar.addWidget(self.mode_mesh)  
  
        self.status_badge = StatusBadge()  
  
        self.restart_button = QPushButton("Restart Stream")  
        self.restart_button.setObjectName("GlassButton")  
  
        header.addLayout(title_block)  
        header.addStretch(1)  
        header.addWidget(mode_container)  
        header.addSpacing(16)  
        header.addWidget(self.status_badge)  
        header.addWidget(self.restart_button)  
  
        content_layout.addLayout(header)  
  
        top_split = QSplitter(Qt.Orientation.Horizontal)  
        top_split.setChildrenCollapsible(False)  
  
        # Left plane  
        video_hw_surface = QFrame()  
        video_hw_surface.setObjectName("Surface")  
        video_hw_layout = QHBoxLayout(video_hw_surface)  
        video_hw_layout.setContentsMargins(24, 24, 24, 24)  
        video_hw_layout.setSpacing(24)  
  
        video_col = QWidget()  
        video_col_layout = QVBoxLayout(video_col)  
        video_col_layout.setContentsMargins(0, 0, 0, 0)  
        video_col_layout.setSpacing(12)  
  
        preview_header = QHBoxLayout()  
        preview_title = QLabel("Live Preview")  
        preview_title.setObjectName("PanelTitle")  
  
        self.preview_mode_badge = QLabel("Face Mesh")  
        self.preview_mode_badge.setObjectName("CardHint")  
  
        preview_header.addWidget(preview_title)  
        preview_header.addStretch(1)  
        preview_header.addWidget(self.preview_mode_badge)  
  
        self.video_pane = VideoPane()  
  
        self.preview_footer = QLabel(f"{FRAME_WIDTH} × {FRAME_HEIGHT} · MediaPipe FaceMesh")  
        self.preview_footer.setObjectName("CardHint")  
        self.preview_footer.setWordWrap(True)  
  
        video_col_layout.addLayout(preview_header)  
        video_col_layout.addWidget(self.video_pane, 1)  
        video_col_layout.addWidget(self.preview_footer)  
  
        hw_col = QWidget()  
        hw_col_layout = QVBoxLayout(hw_col)  
        hw_col_layout.setContentsMargins(0, 0, 0, 0)  
        hw_col_layout.setSpacing(16)  
  
        stats_grid = QGridLayout()  
        stats_grid.setHorizontalSpacing(12)  
        stats_grid.setVerticalSpacing(12)  
  
        self.card_app = MetricTile("Foreground App", "Polling...")  
        self.card_face = MetricTile("Face", "--")  
        self.card_fps = MetricTile("Stream FPS", "--")  
        self.card_dev_cpu = MetricTile("Device CPU", "--", "Waiting for ADB...")  
        self.card_host_system = SecondaryTile("Local System", "--", "NVIDIA SMI unavailable")  
        self.card_au12 = AU12Tile("Live AU12 (Smile)", "0.000", "Waiting for face...")  
  
        stats_grid.addWidget(self.card_app, 0, 0)  
        stats_grid.addWidget(self.card_face, 0, 1)  
        stats_grid.addWidget(self.card_fps, 1, 0, 1, 2)  
        stats_grid.addWidget(self.card_host_system, 2, 0)  
        stats_grid.addWidget(self.card_dev_cpu, 2, 1)  
        stats_grid.addWidget(self.card_au12, 3, 0, 1, 2)  
  
        hw_col_layout.addLayout(stats_grid)  
  
        self.tracking_tile = TrackingPointsTile()  
        hw_col_layout.addWidget(self.tracking_tile)  
        hw_col_layout.addStretch(1)  
  
        video_hw_layout.addWidget(video_col, 5)  
        video_hw_layout.addWidget(hw_col, 4)  
  
        top_split.addWidget(video_hw_surface)  
  
        # Right plane  
        side_scroll = QScrollArea()  
        side_scroll.setWidgetResizable(True)  
        side_scroll.setMinimumWidth(400)  
  
        side_widget = QWidget()  
        side_layout = QVBoxLayout(side_widget)  
        side_layout.setContentsMargins(16, 0, 0, 0)  
        side_layout.setSpacing(16)  
  
        transcript_panel = PanelCard("Latest Transcript")  
        self.transcript_meta = QLabel("Segment: --")  
        self.transcript_meta.setObjectName("CardHint")  
  
        self.transcript_output = QPlainTextEdit()  
        self.transcript_output.setObjectName("TextBox")  
        self.transcript_output.setReadOnly(True)  
        self.transcript_output.setUndoRedoEnabled(False)  
        self.transcript_output.setMaximumHeight(100)  
  
        transcript_panel.body_layout.addWidget(self.transcript_meta)  
        transcript_panel.body_layout.addWidget(self.transcript_output)  
        side_layout.addWidget(transcript_panel)  
  
        semantic_panel = PanelCard("Semantic Evaluation")  
  
        semantic_stats_row = QHBoxLayout()  
        semantic_stats_row.setContentsMargins(0, 0, 0, 0)  
        semantic_stats_row.setSpacing(12)  
  
        self.stat_match = InlineStat("Semantic Match", "--")  
        self.stat_confidence = InlineStat("Confidence", "--")  
  
        semantic_stats_row.addWidget(self.stat_match)  
        semantic_stats_row.addWidget(self.stat_confidence)  
  
        self.reasoning_output = QPlainTextEdit()  
        self.reasoning_output.setObjectName("TextBox")  
        self.reasoning_output.setReadOnly(True)  
        self.reasoning_output.setUndoRedoEnabled(False)  
        self.reasoning_output.setMaximumHeight(80)  
        self.reasoning_output.setPlaceholderText("Waiting for evaluation reasoning...")  
  
        semantic_panel.body_layout.addLayout(semantic_stats_row)  
        semantic_panel.body_layout.addWidget(self.reasoning_output)  
        side_layout.addWidget(semantic_panel)  
  
        reward_panel = PanelCard("Thompson Sampling & Reward")  
  
        self.stat_reward = InlineStat("Reward Summary", "--", compact=True)  
  
        self.experiment_output = QPlainTextEdit()  
        self.experiment_output.setObjectName("TextBox")  
        self.experiment_output.setReadOnly(True)  
        self.experiment_output.setUndoRedoEnabled(False)  
        self.experiment_output.setMaximumHeight(80)  
        self.experiment_output.setPlaceholderText("Waiting for posterior state...")  
  
        reward_panel.body_layout.addWidget(self.stat_reward)  
        reward_panel.body_layout.addWidget(self.experiment_output)  
        side_layout.addWidget(reward_panel)  
  
        acoustic_panel = PanelCard("Acoustic Metrics")  
        self.stat_f0_validity = InlineStat("F0 Validity", "--", compact=True)
        self.stat_perturbation_validity = InlineStat(
            "Perturbation Validity", "--", compact=True
        )
        self.card_pitch = InlineSparkline("Pitch F0", "--", "Hz", "#8B5CF6")  
        self.card_jitter = InlineSparkline("Jitter", "--", "", "#06B6D4")  
        self.card_shimmer = InlineSparkline("Shimmer", "--", "", "#10B981")  

        acoustic_validity_row = QHBoxLayout()
        acoustic_validity_row.setContentsMargins(0, 0, 0, 0)
        acoustic_validity_row.setSpacing(12)
        acoustic_validity_row.addWidget(self.stat_f0_validity)
        acoustic_validity_row.addWidget(self.stat_perturbation_validity)
  
        acoustic_panel.body_layout.addLayout(acoustic_validity_row)
        acoustic_panel.body_layout.addWidget(self.card_pitch)  
  
        acoustic_bottom_grid = QGridLayout()  
        acoustic_bottom_grid.setContentsMargins(0, 0, 0, 0)  
        acoustic_bottom_grid.setHorizontalSpacing(12)  
        acoustic_bottom_grid.setVerticalSpacing(12)  
        acoustic_bottom_grid.addWidget(self.card_jitter, 0, 0)  
        acoustic_bottom_grid.addWidget(self.card_shimmer, 0, 1)  
  
        acoustic_panel.body_layout.addLayout(acoustic_bottom_grid)  
        side_layout.addWidget(acoustic_panel)  
  
        notes_panel = PanelCard("Operator Notes")  
        notes_body = QLabel(  
            "• Orchestrator must be running to populate analytics\n"  
            "• 1 raw · 2 lip points · 3 face mesh\n"  
            "• R = restart UI stream\n"  
            "• C = clear log\n"  
            "• Esc = quit"  
        )  
        notes_body.setObjectName("InfoRow")  
        notes_body.setWordWrap(True)  
  
        notes_panel.body_layout.addWidget(notes_body)  
        side_layout.addWidget(notes_panel)  
        side_layout.addStretch(1)  
  
        side_scroll.setWidget(side_widget)  
  
        top_split.addWidget(side_scroll)
        top_split.setSizes([1040, 420])

        analytics_tab = self._build_analytics_tab()

        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("MainTabs")
        self.tab_widget.addTab(top_split, "Live")
        self.tab_widget.addTab(analytics_tab, "Analytics")

        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.setChildrenCollapsible(False)

        log_surface = QFrame()
        log_surface.setObjectName("Surface")
        log_layout = QVBoxLayout(log_surface)
        log_layout.setContentsMargins(24, 20, 24, 24)
        log_layout.setSpacing(16)

        log_header = QHBoxLayout()
        log_title = QLabel("Session Log")
        log_title.setObjectName("PanelTitle")

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.setObjectName("GlassButton")

        log_header.addWidget(log_title)
        log_header.addStretch(1)
        log_header.addWidget(self.clear_log_button)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setUndoRedoEnabled(False)
        self.log_output.document().setMaximumBlockCount(1200)

        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log_output)

        main_split.addWidget(self.tab_widget)
        main_split.addWidget(log_surface)
        main_split.setSizes([850, 150])
  
        content_layout.addWidget(main_split, 1)  
        main_layout.addWidget(content_widget, 1)  
        self.root_layout.addWidget(self.main_frame)  
  
        self.size_grip = QSizeGrip(self.main_frame)  
        main_layout.addWidget(self.size_grip, 0, Qt.AlignBottom | Qt.AlignRight)  
  
        self.shortcut_restart = QShortcut(QKeySequence("R"), self)  
        self.shortcut_clear = QShortcut(QKeySequence("C"), self)  
        self.shortcut_raw = QShortcut(QKeySequence("1"), self)  
        self.shortcut_lips = QShortcut(QKeySequence("2"), self)  
        self.shortcut_mesh = QShortcut(QKeySequence("3"), self)  
        self.shortcut_escape = QShortcut(QKeySequence("Esc"), self)  
  
        for shortcut in (  
            self.shortcut_restart,  
            self.shortcut_clear,  
            self.shortcut_raw,  
            self.shortcut_lips,  
            self.shortcut_mesh,  
            self.shortcut_escape,  
        ):  
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)  
  
    def _build_analytics_tab(self) -> QWidget:
        container = QScrollArea()
        container.setWidgetResizable(True)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(24, 24, 24, 24)
        inner_layout.setSpacing(16)

        # Stimulus control bar
        stimulus_bar = QHBoxLayout()
        stimulus_bar.setContentsMargins(0, 0, 0, 0)
        stimulus_bar.setSpacing(12)

        self.stimulus_button = QPushButton("Inject Stimulus")
        self.stimulus_button.setObjectName("GlassButton")

        self.stimulus_custom_button = QPushButton("Inject Custom Line…")
        self.stimulus_custom_button.setObjectName("GlassButton")

        self.stimulus_status_label = QLabel("")
        self.stimulus_status_label.setObjectName("CardHint")
        self.stimulus_status_label.setWordWrap(True)

        stimulus_bar.addWidget(self.stimulus_button)
        stimulus_bar.addWidget(self.stimulus_custom_button)
        stimulus_bar.addWidget(self.stimulus_status_label, 1)

        inner_layout.addLayout(stimulus_bar)

        # Physiology panel — two columns (streamer / operator)
        physio_panel = PanelCard("Physiological State")

        physio_row = QHBoxLayout()
        physio_row.setContentsMargins(0, 0, 0, 0)
        physio_row.setSpacing(12)

        self.streamer_rmssd = InlineStat("Streamer RMSSD", "--", "ms")
        self.streamer_hr = InlineStat("Streamer HR", "--", "bpm")
        self.streamer_freshness = InlineStat("Streamer Freshness", "--", "s")

        self.operator_rmssd = InlineStat("Operator RMSSD", "--", "ms")
        self.operator_hr = InlineStat("Operator HR", "--", "bpm")
        self.operator_freshness = InlineStat("Operator Freshness", "--", "s")

        streamer_col = QVBoxLayout()
        streamer_col.setSpacing(8)
        self.streamer_badge = StatusBadge()
        self.streamer_badge.set_status("NO DATA", "neutral")
        self.streamer_provider_label = QLabel("Provider: --")
        self.streamer_provider_label.setObjectName("CardHint")
        streamer_col.addWidget(self.streamer_badge)
        streamer_col.addWidget(self.streamer_rmssd)
        streamer_col.addWidget(self.streamer_hr)
        streamer_col.addWidget(self.streamer_freshness)
        streamer_col.addWidget(self.streamer_provider_label)

        operator_col = QVBoxLayout()
        operator_col.setSpacing(8)
        self.operator_badge = StatusBadge()
        self.operator_badge.set_status("NO DATA", "neutral")
        self.operator_provider_label = QLabel("Provider: --")
        self.operator_provider_label.setObjectName("CardHint")
        operator_col.addWidget(self.operator_badge)
        operator_col.addWidget(self.operator_rmssd)
        operator_col.addWidget(self.operator_hr)
        operator_col.addWidget(self.operator_freshness)
        operator_col.addWidget(self.operator_provider_label)

        physio_row.addLayout(streamer_col, 1)
        physio_row.addLayout(operator_col, 1)

        physio_panel.body_layout.addLayout(physio_row)
        inner_layout.addWidget(physio_panel)

        # Co-Modulation Index panel
        comod_panel = PanelCard("Co-Modulation Index (5-min rolling)")

        comod_top_row = QHBoxLayout()
        comod_top_row.setContentsMargins(0, 0, 0, 0)
        comod_top_row.setSpacing(12)

        self.comod_value = InlineStat("Latest CMI", "--")
        self.comod_coverage = InlineStat("Coverage Ratio", "--")
        self.comod_pairs = InlineStat("Paired Obs.", "--")

        comod_top_row.addWidget(self.comod_value)
        comod_top_row.addWidget(self.comod_coverage)
        comod_top_row.addWidget(self.comod_pairs)

        self.comod_sparkline = SparklineWidget("#F59E0B")
        self.comod_sparkline.setMinimumHeight(120)

        self.comod_hint_label = QLabel("Awaiting alignment — insufficient paired observations")
        self.comod_hint_label.setObjectName("CardHint")
        self.comod_hint_label.setWordWrap(True)
        self.comod_hint_label.setVisible(False)

        comod_panel.body_layout.addLayout(comod_top_row)
        comod_panel.body_layout.addWidget(self.comod_sparkline)
        comod_panel.body_layout.addWidget(self.comod_hint_label)
        inner_layout.addWidget(comod_panel)

        # Encounter reward summary (per-arm)
        summary_panel = PanelCard(f"Encounter Reward Summary · {DEFAULT_EXPERIMENT_ID}")

        self.encounter_table = QTableWidget(0, 6)
        self.encounter_table.setHorizontalHeaderLabels(
            ["Arm", "Encounters", "Valid", "Avg Reward", "Gate Rate", "Avg Frames"]
        )
        self.encounter_table.verticalHeader().setVisible(False)
        self.encounter_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.encounter_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.encounter_table.setMinimumHeight(160)
        header = self.encounter_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        summary_panel.body_layout.addWidget(self.encounter_table)
        inner_layout.addWidget(summary_panel)

        inner_layout.addStretch(1)

        container.setWidget(inner)
        return container

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.WindowStateChange:  
            maximized = self.isMaximized()  
  
            if maximized:  
                self.root_layout.setContentsMargins(0, 0, 0, 0)  
                self.main_frame.setProperty("maximized", "true")  
                self.shadow.setEnabled(False)  
                self.size_grip.hide()  
            else:  
                self.root_layout.setContentsMargins(24, 24, 24, 24)  
                self.main_frame.setProperty("maximized", "false")  
                self.shadow.setEnabled(True)  
                self.size_grip.show()  
  
            self.main_frame.style().unpolish(self.main_frame)  
            self.main_frame.style().polish(self.main_frame)  
            self.main_frame.update()  
  
        super().changeEvent(event)  
  
    def _connect_signals(self) -> None:
        self.restart_button.clicked.connect(self._manual_restart)
        self.clear_log_button.clicked.connect(self.log_output.clear)
        self.stimulus_button.clicked.connect(self._inject_stimulus_default)
        self.stimulus_custom_button.clicked.connect(self._inject_stimulus_custom)
  
        self.shortcut_restart.activated.connect(self._manual_restart)  
        self.shortcut_clear.activated.connect(self.log_output.clear)  
        self.shortcut_raw.activated.connect(lambda: self._set_overlay_mode(OVERLAY_RAW))  
        self.shortcut_lips.activated.connect(lambda: self._set_overlay_mode(OVERLAY_LIPS))  
        self.shortcut_mesh.activated.connect(lambda: self._set_overlay_mode(OVERLAY_FULL))  
        self.shortcut_escape.activated.connect(self.close)  
  
        self.mode_raw.clicked.connect(lambda: self._set_overlay_mode(OVERLAY_RAW))  
        self.mode_lips.clicked.connect(lambda: self._set_overlay_mode(OVERLAY_LIPS))  
        self.mode_mesh.clicked.connect(lambda: self._set_overlay_mode(OVERLAY_FULL))  
  
        self.thread.frame_ready.connect(self.video_pane.set_frame)  
        self.thread.status_changed.connect(self._set_status)  
        self.thread.metrics_changed.connect(self._update_metrics)  
        self.thread.log_message.connect(self._append_log)  
  
        self.analytics_thread.snapshot_ready.connect(self._update_operator_snapshot)  
        self.analytics_thread.log_message.connect(self._append_log)  
  
        self.hw_thread.hw_stats_ready.connect(self._update_hw_stats)  
        self.hw_thread.log_message.connect(self._append_log)  
  
    def _set_overlay_mode(self, mode: str, log_message: bool = True) -> None:  
        previous = self.thread.get_overlay_mode()  
        self.thread.set_overlay_mode(mode)  
  
        if mode == OVERLAY_RAW:  
            self.mode_raw.setChecked(True)  
        elif mode == OVERLAY_LIPS:  
            self.mode_lips.setChecked(True)  
        else:  
            self.mode_mesh.setChecked(True)  
  
        label = mode_label(mode)  
        set_text_if_changed(self.preview_mode_badge, label)  
  
        if log_message and previous != mode:  
            self._append_log(f"Overlay mode set to {label}")  
  
    def _manual_restart(self) -> None:  
        self._append_log("Manual restart requested from UI")  
        self.thread.request_restart()  
  
    def _set_status(self, text: str, level: str) -> None:  
        self.status_badge.set_status(text, level)  
  
    def _update_hw_stats(self, stats: dict[str, str]) -> None:  
        host_cpu = stats.get("host_cpu", "--")  
        host_gpu = stats.get("host_gpu", "--")  
        self.card_host_system.set_value(f"{host_cpu} CPU · {host_gpu} GPU")  
        self.card_host_system.set_hint(stats.get("host_gpu_hint", "NVIDIA SMI unavailable"))  
  
        self.card_dev_cpu.set_value(stats.get("dev_cpu", "--"))  
        self.card_dev_cpu.set_hint(stats.get("dev_cpu_hint", "Waiting for ADB..."))  
  
        self.card_app.set_value(stats.get("fg_app", "--"))  
        self.card_app.set_hint(stats.get("fg_app_hint", ""))  
  
    def _update_metrics(self, metrics: dict[str, Any]) -> None:  
        self.card_face.set_value(metrics.get("face", "--"))  
  
        self.card_fps.set_value(metrics.get("fps", "--"))  
        self.card_fps.set_hint(  
            f"min: {metrics.get('fps_min', '--')}  |  "  
            f"avg: {metrics.get('fps_avg', '--')}  |  "  
            f"max: {metrics.get('fps_max', '--')}"  
        )  
  
        reconnects = str(metrics.get("reconnects", "0"))  
        overlay = str(metrics.get("overlay", mode_label(self.thread.get_overlay_mode())))  
  
        device_id = self.hw_thread.get_latest_device_identity()  
        device_str = f" · {device_id}" if device_id else ""  
  
        footer_text = (  
            f"{FRAME_WIDTH} × {FRAME_HEIGHT} · MediaPipe FaceMesh · Overlay: {overlay}  |  "  
            f"Source: {STREAM_SERVICE} ({VIDEO_PIPE}){device_str}  |  "  
            f"Reconnects: {reconnects}"  
        )  
        set_text_if_changed(self.preview_mode_badge, overlay)  
        set_text_if_changed(self.preview_footer, footer_text)  
  
        self.tracking_tile.set_values(  
            metrics.get("left_lip", "--"),  
            metrics.get("right_lip", "--"),  
            metrics.get("mouth_span", "--"),  
        )  
  
        try:  
            au12_numeric = float(metrics.get("au12_val", 0.0) or 0.0)  
        except (TypeError, ValueError):  
            au12_numeric = 0.0  
  
        self.card_au12.set_value(f"{au12_numeric:.3f}", au12_numeric)  
        self.card_au12.set_hint(metrics.get("au12_status", ""))  
  
    def _update_operator_snapshot(self, snapshot: dict[str, Any]) -> None:  
        metrics = snapshot.get("metrics") or {}  
        history = snapshot.get("metrics_history") or []  
        transcript = snapshot.get("transcript") or {}  
        evaluation = snapshot.get("evaluation") or {}  
        encounter = snapshot.get("encounter") or {}  
        experiments = snapshot.get("experiments") or []  
  
        transcript_text = str(  
            pick(transcript, "transcription", "transcript", "text", default="--")  
        )  
        transcript_segment = pick(transcript, "segment_id", default="--")  
        transcript_time = pick(transcript, "created_at", "timestamp_utc", default="--")  
  
        set_text_if_changed(  
            self.transcript_meta,  
            f"Segment: {transcript_segment} · {transcript_time}",  
        )  
        set_plaintext_if_changed(self.transcript_output, transcript_text)  
  
        self.stat_f0_validity.set_value(
            format_validity_pair(metrics, "f0_valid_measure", "f0_valid_baseline")
        )
        self.stat_perturbation_validity.set_value(
            format_validity_pair(
                metrics,
                "perturbation_valid_measure",
                "perturbation_valid_baseline",
            )
        )

        pitch_value = first_present(metrics, "f0_mean_measure_hz", "pitch_f0")
        jitter_value = first_present(metrics, "jitter_mean_measure", "jitter")
        shimmer_value = first_present(metrics, "shimmer_mean_measure", "shimmer")

        f0_delta_hint = format_delta_line(
            first_present(metrics, "f0_delta_semitones"), 2, " st"
        )
        jitter_delta_hint = format_delta_line(first_present(metrics, "jitter_delta"), 4)
        shimmer_delta_hint = format_delta_line(first_present(metrics, "shimmer_delta"), 4)

        self.card_pitch.set_value(format_optional_float(pitch_value, 2))
        self.card_pitch.set_hint(f"Hz · {f0_delta_hint}" if f0_delta_hint else "Hz")
        self.card_jitter.set_value(format_optional_float(jitter_value, 4))
        self.card_jitter.set_hint(jitter_delta_hint)
        self.card_shimmer.set_value(format_optional_float(shimmer_value, 4))
        self.card_shimmer.set_hint(shimmer_delta_hint)

        ordered_history = list(reversed(history if isinstance(history, list) else []))
        pitches: list[float] = []
        jitters: list[float] = []
        shimmers: list[float] = []

        for row in ordered_history:
            if not isinstance(row, dict):
                continue
            for target, keys in (
                (pitches, ("f0_mean_measure_hz", "pitch_f0")),
                (jitters, ("jitter_mean_measure", "jitter")),
                (shimmers, ("shimmer_mean_measure", "shimmer")),
            ):
                try:
                    number = float(first_present(row, *keys))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    target.append(number)

        self.card_pitch.set_values(pitches)
        self.card_jitter.set_values(jitters)
        self.card_shimmer.set_values(shimmers)
  
  
        is_match = pick(evaluation, "is_match", default=None)  
        if isinstance(is_match, bool):  
            self.stat_match.set_value("YES" if is_match else "NO")  
        else:  
            self.stat_match.set_value(str(is_match if is_match is not None else "--"))  
  
        confidence = pick(evaluation, "confidence_score", default="--")  
        self.stat_confidence.set_value(format_numeric_or_text(confidence, 2))  
  
        reasoning_text = str(pick(evaluation, "reasoning", default="Waiting for evaluation..."))  
        set_plaintext_if_changed(self.reasoning_output, reasoning_text)  
  
        reward = format_if_number(deep_find(encounter, "gated_reward", default="--"), 3)  
        p90 = format_if_number(deep_find(encounter, "p90_intensity", default="--"), 3)  
        gate = deep_find(encounter, "semantic_gate", default="--")  
        frames = deep_find(encounter, "n_frames_in_window", default="--")  
  
        reward_text = f"r = {reward}  ·  p90 = {p90}  ·  gate = {gate}  ·  frames = {frames}"  
        self.stat_reward.set_value(reward_text)  
  
        exp_lines: list[str] = []  
        for row in experiments:  
            try:  
                arm = row.get("arm", "unknown")  
                alpha = float(row.get("alpha_param", 1.0))  
                beta = float(row.get("beta_param", 1.0))  
                mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0  
                exp_lines.append(f"{arm}: α={alpha:.1f}, β={beta:.1f} (μ={mean:.2f})")  
            except (TypeError, ValueError):  
                continue  
  
        set_plaintext_if_changed(self.experiment_output, "\n".join(exp_lines))

        streamer_physio = snapshot.get("streamer_physio") or {}
        operator_physio = snapshot.get("operator_physio") or {}
        comodulation = snapshot.get("comodulation") or []
        encounter_summary = snapshot.get("encounter_summary") or []

        self._update_physio_column(
            streamer_physio,
            self.streamer_badge,
            self.streamer_rmssd,
            self.streamer_hr,
            self.streamer_freshness,
            self.streamer_provider_label,
        )
        self._update_physio_column(
            operator_physio,
            self.operator_badge,
            self.operator_rmssd,
            self.operator_hr,
            self.operator_freshness,
            self.operator_provider_label,
        )

        self._update_comodulation(comodulation)
        self._update_encounter_summary(encounter_summary)

    def _update_physio_column(
        self,
        row: dict[str, Any],
        badge: StatusBadge,
        rmssd_tile: InlineStat,
        hr_tile: InlineStat,
        freshness_tile: InlineStat,
        provider_label: QLabel,
    ) -> None:
        if not row:
            badge.set_status("NO DATA", "neutral")
            rmssd_tile.set_value("--")
            hr_tile.set_value("--")
            freshness_tile.set_value("--")
            set_text_if_changed(provider_label, "Provider: --")
            return

        is_stale = bool(row.get("is_stale"))
        if is_stale:
            badge.set_status("STALE", "warn")
        else:
            badge.set_status("FRESH", "ok")

        rmssd_tile.set_value(format_optional_float(row.get("rmssd_ms"), 1))
        hr = row.get("heart_rate_bpm")
        hr_tile.set_value(str(hr) if hr is not None else "--")
        freshness_tile.set_value(format_optional_float(row.get("freshness_s"), 1))
        set_text_if_changed(provider_label, f"Provider: {row.get('provider', '--')}")

    def _update_comodulation(self, history: list[dict[str, Any]]) -> None:
        values: list[float] = []
        for row in history:
            cmi = row.get("co_modulation_index")
            if cmi is None:
                continue
            try:
                values.append(float(cmi))
            except (TypeError, ValueError):
                continue
        self.comod_sparkline.set_values(values)

        latest = history[-1] if history else {}
        cmi_value = latest.get("co_modulation_index") if latest else None
        coverage = latest.get("coverage_ratio") if latest else None
        pairs = latest.get("n_paired_observations") if latest else None

        if cmi_value is None:
            self.comod_value.set_value("--")
            self.comod_hint_label.setVisible(bool(latest))
        else:
            self.comod_value.set_value(format_optional_float(cmi_value, 3))
            self.comod_hint_label.setVisible(False)

        self.comod_coverage.set_value(format_optional_float(coverage, 2))
        self.comod_pairs.set_value(str(pairs) if pairs is not None else "--")

    def _update_encounter_summary(self, rows: list[dict[str, Any]]) -> None:
        self.encounter_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            cells = [
                str(row.get("arm", "--")),
                str(row.get("encounter_count", 0)),
                str(row.get("valid_count", 0)),
                format_optional_float(row.get("avg_reward"), 3),
                format_optional_float(row.get("gate_rate"), 2),
                format_optional_float(row.get("avg_frames"), 1),
            ]
            for col_idx, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.encounter_table.setItem(row_idx, col_idx, item)

    def _inject_stimulus_default(self) -> None:
        self._post_stimulus(None)

    def _inject_stimulus_custom(self) -> None:
        text, ok = QInputDialog.getText(
            self,
            "Inject Custom Stimulus",
            "Line text (leave blank for default):",
        )
        if not ok:
            return
        trimmed = text.strip() or None
        self._post_stimulus(trimmed)

    def _post_stimulus(self, line_text: str | None) -> None:
        # WRITE PATH — the only place this module talks to FastAPI. The API
        # owns the Redis pub/sub publish; do not bypass it from the GUI.
        url = f"{API_BASE}/api/v1/stimulus"
        payload = json.dumps({"line_text": line_text} if line_text else {}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            msg = f"Stimulus inject failed ({exc.code}): {detail[:200]}"
            set_text_if_changed(self.stimulus_status_label, msg)
            self._append_log(msg)
            return
        except urllib.error.URLError as exc:
            msg = f"Stimulus inject unreachable: {exc.reason}"
            set_text_if_changed(self.stimulus_status_label, msg)
            self._append_log(msg)
            return

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            decoded = {"status": "unknown", "raw": body[:120]}

        status = decoded.get("status", "unknown")
        receivers = decoded.get("receivers")
        summary = f"Stimulus {status}"
        if receivers is not None:
            summary += f" · receivers={receivers}"
        warning = decoded.get("warning")
        if warning:
            summary += f" · {warning}"
        set_text_if_changed(self.stimulus_status_label, summary)
        self._append_log(summary)

    def _append_log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")  
        self.log_output.appendPlainText(f"[{ts}] {message}")  
  
    def closeEvent(self, event: QCloseEvent) -> None:  
        self._append_log("Shutting down...")  
        self.thread.stop()  
        self.analytics_thread.stop()  
        self.hw_thread.stop()  
  
        self.thread.wait(5000)  
        self.analytics_thread.wait(3000)  
        self.hw_thread.wait(3000)  
  
        super().closeEvent(event)  
  
  
def main() -> int:  
    app = QApplication(sys.argv)  
    app.setApplicationName(APP_NAME)  
    app.setStyle("Fusion")  
  
    window = MainWindow()  
    window.show()  
  
    return app.exec()  
  
  
if __name__ == "__main__":  
    raise SystemExit(main())  