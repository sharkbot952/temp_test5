# test_version
import os
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# =========================================
# 設定（フォルダ固定）
# =========================================
BASE_DIR = str(Path(__file__).parent.joinpath("data").resolve())
PRED_DIR = "pred"
OBS_DIR = "obs"
CORR_DIR = "corr"
CMEM_DIR = "cmem"
CMEM_THETAO_DIR = "thetao"
CMEM_CHL_DIR = "chl"

# 固定パラメータ
RECENT_DAYS = 7           
OUTLIER_TH = 4.0          
OUTLIER_TH_OBS = 2.0      
OBS_MATCH_TOL_MIN = 60    
CORR_MATCH_TOL_MIN = 60   
TEMP_MIN, TEMP_MAX = -2.0, 40.0
PHYS_MIN, PHYS_MAX = -1.5, 35.0
HIGH_TEMP_TH = 22.0       # コメント用
RANGE_STABLE = 0.5
DELTA_THRESH = 0.3
DISPLAY_MODE = "arrow"

WEEK_WINDOW_FORWARD = True  # True: 今日→先7日（計8日）、False: 過去7日→今日（計8日）

def pjoin(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))

# =========================================
# ユーティリティ
# =========================================
def _pick_series_corr_then_pred(g: pd.DataFrame) -> Optional[pd.Series]:
    cand = None
    if "corr_temp" in g.columns:
        c = pd.to_numeric(g["corr_temp"], errors="coerce")
        if c.notna().sum() >= 1:
            cand = c
    if cand is None and "pred_temp" in g.columns:
        p = pd.to_numeric(g["pred_temp"], errors="coerce")
        if p.notna().sum() >= 1:
            cand = p
    return cand

def utc_to_jst_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def jst_to_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def safe_merge_asof_by_depth_keep_left(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
    right_value_cols: List[str],
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    out_list: List[pd.DataFrame] = []
    left_depths = sorted(set(left["depth_m"].dropna().unique()))
    for d in left_depths:
        l = left[left["depth_m"] == d].sort_values("datetime")
        r = right[right["depth_m"] == d].sort_values("datetime")[["datetime", "depth_m"] + right_value_cols]
        if l.empty:
            continue
        if r.empty:
            pad = l.copy()
            for c in right_value_cols:
                pad[c] = np.nan
            out_list.append(pad)
        else:
            merged = pd.merge_asof(
                l, r, on="datetime", by="depth_m",
                tolerance=tolerance, direction="nearest", suffixes=suffixes
            )
            out_list.append(merged)
    if not out_list:
        out = left.copy()
        for c in right_value_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out
    return pd.concat(out_list, ignore_index=True)

def _detect_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        if c.lower() in [k.lower() for k in keywords]:
            return c
    norm = {c: c.lower().replace("_", "") for c in cols}
    for c, n in norm.items():
        ok = all(k.lower().replace("_", "") in n for k in keywords)
        if ok:
            return c
    return None

def to_rgba(color: str, alpha: float = 0.18) -> str:
    if not isinstance(color, str) or not color:
        return f"rgba(0,150,0,{alpha})"
    c = color.strip().lower()
    if c.startswith("rgba(") and c.endswith(")"):
        try:
            nums = c[5:-1].split(",")
            r, g, b = [int(float(x)) for x in nums[:3]]
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"
    if c.startswith("rgb(") and c.endswith(")"):
        try:
            r, g, b = [int(float(x)) for x in c[4:-1].split(",")[:3]]
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"
    if c.startswith("#"):
        h = c.lstrip("#")
        try:
            if len(h) == 3:
                r = int(h[0]*2, 16); g = int(h[1]*2, 16); b = int(h[2]*2, 16)
            elif len(h) == 6:
                r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
            else:
                return f"rgba(0,150,0,{alpha})"
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"
    return c

# ---- キャッシュ無効化用：ファイル指紋 ----
def file_fingerprint(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    try:
        st_ = p.stat()
        return f"mtime:{int(st_.st_mtime)}:size:{st_.st_size}"
    except Exception:
        return "exists"

def obs_fingerprint(base_dir: str, obs_dir: str, filename: str) -> str:
    path = os.path.normpath(os.path.join(base_dir, obs_dir, filename))
    return file_fingerprint(path)

# =========================================
# ローダ（fp をキーに追加）
# =========================================
@st.cache_data(show_spinner=False)
def load_pred(filename: str, fp: str = "") -> pd.DataFrame:
    path = pjoin(BASE_DIR, PRED_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = utc_to_jst_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    df = df.rename(columns={"Temp": "pred_temp"})
    if ("U" in df.columns) and ("V" in df.columns):
        df["U"] = pd.to_numeric(df["U"], errors="coerce")
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["Speed"] = np.sqrt(np.square(df["U"]) + np.square(df["V"]))
        df["Direction_deg"] = (np.degrees(np.arctan2(df["U"], df["V"])) + 360.0) % 360.0
    df = df.dropna(subset=["datetime", "depth_m"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df

@st.cache_data(show_spinner=False)
def load_corr_for(filename: str, fp: str = "") -> pd.DataFrame:
    name, ext = os.path.splitext(filename)
    corr_filename = f"{name}_corr{ext}"
    path = pjoin(BASE_DIR, CORR_DIR, corr_filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = jst_to_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    corr_col = _detect_column(df, ["corr", "temp"]) or ("CorrTemp" if "CorrTemp" in df.columns else None)
    if corr_col is None:
        corr_col = "Temp" if "Temp" in df.columns else None
    if corr_col is None:
        return pd.DataFrame()
    low_col  = _detect_column(df, ["corr", "low"])  or ("CorrLow"  if "CorrLow"  in df.columns else None)
    high_col = _detect_column(df, ["corr", "high"]) or ("CorrHigh" if "CorrHigh" in df.columns else None)
    rename_map = {corr_col: "corr_temp"}
    if low_col:  rename_map[low_col]  = "corr_low"
    if high_col: rename_map[high_col] = "corr_high"
    df = df.rename(columns=rename_map)
    keep = ["datetime", "depth_m", "corr_temp"]
    if "corr_low" in df.columns:  keep.append("corr_low")
    if "corr_high" in df.columns: keep.append("corr_high")
    df = df[keep].dropna(subset=["datetime", "depth_m", "corr_temp"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df

@st.cache_data(show_spinner=False)
def load_obs_for(filename: str, fp: str = "") -> pd.DataFrame:
    path = pjoin(BASE_DIR, OBS_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = jst_to_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    df = df.rename(columns={"Temp": "obs_temp"})
    df = df.dropna(subset=["datetime", "depth_m"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df
# =========================================
# CMEM ローダ（thetao / chl）
# =========================================

def _extract_site_from_filename(fname: str, prefix: str) -> str:
    base = os.path.basename(fname)
    if base.lower().startswith(prefix.lower()) and base.lower().endswith('.csv'):
        return base[len(prefix):-4]
    return ""

def list_cmem_sites() -> List[str]:
    thetao_folder = pjoin(BASE_DIR, CMEM_DIR, CMEM_THETAO_DIR)
    chl_folder = pjoin(BASE_DIR, CMEM_DIR, CMEM_CHL_DIR)
    thetao_sites, chl_sites = set(), set()
    if os.path.exists(thetao_folder):
        for f in os.listdir(thetao_folder):
            s = _extract_site_from_filename(f, 'thetao_')
            if s:
                thetao_sites.add(s)
    if os.path.exists(chl_folder):
        for f in os.listdir(chl_folder):
            s = _extract_site_from_filename(f, 'chl_')
            if s:
                chl_sites.add(s)
    return sorted(thetao_sites.intersection(chl_sites))

@st.cache_data(show_spinner=False)
def load_cmem_thetao(site: str, fp: str = "") -> pd.DataFrame:
    path = pjoin(BASE_DIR, CMEM_DIR, CMEM_THETAO_DIR, f"thetao_{site}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    if 'flag' in df.columns:
        df = df[pd.to_numeric(df['flag'], errors='coerce') == 1]
    df['datetime'] = pd.to_datetime(df.get('Date'), errors='coerce')
    df['depth_m'] = pd.to_numeric(df.get('Depth'), errors='coerce').round(0).astype('Int64')
    val_col = 'Temp' if 'Temp' in df.columns else ('thetao' if 'thetao' in df.columns else None)
    if val_col is None:
        return pd.DataFrame()
    df = df.rename(columns={val_col: 'thetao'})
    df = df.dropna(subset=['datetime','depth_m','thetao']).copy()
    df['date_day'] = df['datetime'].dt.date
    return df

@st.cache_data(show_spinner=False)
def load_cmem_chl(site: str, fp: str = "") -> pd.DataFrame:
    path = pjoin(BASE_DIR, CMEM_DIR, CMEM_CHL_DIR, f"chl_{site}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    if 'flag' in df.columns:
        df = df[pd.to_numeric(df['flag'], errors='coerce') == 1]
    df['datetime'] = pd.to_datetime(df.get('Date'), errors='coerce')
    df['depth_m'] = pd.to_numeric(df.get('Depth'), errors='coerce').round(0).astype('Int64')
    val_col = 'chl' if 'chl' in df.columns else ('Temp' if 'Temp' in df.columns else None)
    if val_col is None:
        return pd.DataFrame()
    df = df.rename(columns={val_col: 'chl'})
    df = df.dropna(subset=['datetime','depth_m','chl']).copy()
    df['date_day'] = df['datetime'].dt.date
    return df

def add_corr(df_pred: pd.DataFrame, df_corr: pd.DataFrame) -> pd.DataFrame:
    if df_pred.empty or df_corr.empty:
        out = df_pred.copy()
        if "corr_temp" not in out.columns:
            out["corr_temp"] = np.nan
        if "corr_low" not in out.columns:
            out["corr_low"] = np.nan
        if "corr_high" not in out.columns:
            out["corr_high"] = np.nan
        return out

    tol = pd.Timedelta(minutes=CORR_MATCH_TOL_MIN)
    right_cols = ["corr_temp"]
    if "corr_low" in df_corr.columns: right_cols.append("corr_low")
    if "corr_high" in df_corr.columns: right_cols.append("corr_high")

    right = df_corr.sort_values(["depth_m", "datetime"])[["datetime", "depth_m"] + right_cols]
    left = df_pred.sort_values(["depth_m", "datetime"]).copy()

    merged = safe_merge_asof_by_depth_keep_left(
        left, right, tol, right_value_cols=right_cols, suffixes=("", "")
    )
    return merged

# ---- 余白圧縮CSS（強化版） ----
def inject_compact_css():
    compact_css = """
    <style>
    /* 1) ヘッダー・フッター・メニューの非表示 */
    [data-testid="stHeader"], header, .stAppHeader { display: none !important; height: 0 !important; }
    footer, #MainMenu, .viewerBadge_container__1QSob { display: none !important; }

    /* 2) 全体コンテナのパディングを極限まで詰める */
    .block-container { padding-top: 0px !important; padding-bottom: 0px !important; }
    
    /* 3) ウィジェットの「ラベル」が占有する領域を完全にゼロにする */
    div[data-testid="stWidgetLabel"] {
      display: none !important;
      height: 0 !important;
      margin: 0 !important;
      padding: 0 !important;
    }

    /* 4) 要素（各ウィジェット）の上下余白を最小化 */
    div[data-testid="stElementContainer"] {
      margin-top: -4px !important;  /* ネガティブマージンで隙間を詰める */
      margin-bottom: 0px !important;
    }

    /* 5) 水平・垂直方向のブロック間のギャップを最小化 */
    div[data-testid="stVerticalBlock"] {
      gap: 0px !important;
    }
    div[data-testid="stHorizontalBlock"] {
      gap: 4px !important;
      margin-top: 0px !important;
    }

    /* 6) 分割コントロール（Segmented Control）自体の高さを抑える */
    div[data-testid="stSegmentedControl"] {
      margin-top: 0px !important;
      margin-bottom: 2px !important;
    }

    /* 7) Markdown（説明文など）の上下余白調整 */
    .stMarkdown p { margin-top: 2px !important; margin-bottom: 2px !important; line-height: 1.2 !important; }

    /* 8) Plotlyグラフの上部マージン調整 */
    .js-plotly-plot { margin-top: 0px !important; }
    

/* === 追加：選択タブ（segmented_control）間の縦スペースを強制的に詰める === */
div[data-testid="stSegmentedControl"]{
  margin-top: 0px !important;
  margin-bottom: 0px !important;
  padding-top: 0px !important;
  padding-bottom: 0px !important;
}
div[data-testid="stSegmentedControl"] + div[data-testid="stSegmentedControl"]{
  margin-top: -6px !important;
}
div[data-testid="stWidget"]{
  margin-bottom: 0px !important;
  padding-bottom: 0px !important;
}

/* === 追加：tabs の上下余白を詰める（st.tabs 用） === */
div[data-testid="stTabs"]{ margin-top: 0px !important; margin-bottom: 0px !important; }
div[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap: 2px !important; }
div[data-testid="stTabs"] button[data-baseweb="tab"]{ padding: 2px 8px !important; margin: 0px !important; }

</style>
    """
    st_html(compact_css, height=0)

# =========================================
# カレンダー表示の部品（補正値＆矢印を表示）
# =========================================
HEAD_LENGTH_RATIO = 0.55
HEAD_HALF_HEIGHT_RATIO = 0.35
SHAFT_WIDTH_PX = 4.0

def get_arrow_svg(direction_deg, speed_mps):
    if pd.isna(speed_mps) or pd.isna(direction_deg):
        return ""
    css_angle = (direction_deg - 90) % 360
    def _style(s):
        if np.isnan(s): return 18, "#CCCCCC"
        speed_kt = s * 1.94384
        if speed_kt < 1.0: return 18, "#0000FF"
        elif speed_kt < 2.0: return 22, "#FFC107"
        else: return 26, "#FF0000"
    size, color = _style(speed_mps)
    head_length = size * globals().get("HEAD_LENGTH_RATIO", 0.55)
    head_half_h = size * globals().get("HEAD_HALF_HEIGHT_RATIO", 0.35)
    line_end = size - head_length
    return f"""
<svg width="{size}" height="{size}" style="display:block;margin:0 auto;transform:rotate({css_angle}deg);">
  <line x1="4" y1="{size/2}" x2="{line_end}" y2="{size/2}"
        stroke="{color}" stroke-width="{SHAFT_WIDTH_PX}" stroke-linecap="round"/>
  <polygon points="{line_end},{size/2 - head_half_h} {size},{size/2} {line_end},{size/2 + head_half_h}"
           fill="{color}"/>
</svg>
""".strip()

def get_color(temp: float, t_min: float = 0.0, t_max: float = 25.0) -> str:
    if pd.isna(temp): return "rgba(220,220,220,0.4)"
    ratio = (float(temp) - t_min) / (t_max - t_min)
    ratio = max(0, min(1, ratio))
    if ratio < 0.5:
        r = int(240 * ratio * 2); g = int(240 * ratio * 2); b = 240
    else:
        r = 240; g = int(240 * (1 - (ratio - 0.5) * 2)); b = int(240 * (1 - (ratio - 0.5) * 2))
    return f"rgba({r},{g},{b},0.4)"

def get_calendar_css(max_h_vh: int = 65) -> str:
    return f"""
    <style>
    .calendar-scroll-container {{
      overflow-x: auto; overflow-y: auto;
      max-height: {max_h_vh}vh; max-width: 100%;
      -webkit-overflow-scrolling: touch;
      border: 1px solid #e5e5e5; border-radius: 8px;
      isolation: isolate;
    }}
    .calendar-table {{
      border-collapse: separate; border-spacing: 0;
      width: max-content; min-width: 640px; font-size: 14px;
    }}
    .calendar-table th, .calendar-table td {{
      padding: 6px 10px;
      border-bottom: 1px solid #eee;
      text-align: center;
      white-space: nowrap;
    }}
    thead th {{
      position: sticky; top: 0;
      background: #fafafa; z-index: 2;
    }}
    .calendar-table tbody th.depth-cell,
    .calendar-table tbody td.depth-cell {{
      position: sticky; left: 0;
      background: #f7f7f7; z-index: 3;
      min-width: 56px; text-align: center; font-weight: 700 !important;
    }}
    thead th:first-child {{
      position: sticky; left: 0; top: 0;
      background: #f0f0f0; z-index: 4; min-width: 56px; text-align: center; font-weight: 700;
    }}
    .calendar-table .pred-small {{ font-size: 12px; color: #555; }}
    </style>
    """.strip()

def correction_effective(
    temp_pred: Optional[float],
    temp_corr: Optional[float],
    temp_obs: Optional[float] = None
) -> bool:
    if temp_pred is None or pd.isna(temp_pred): return False
    if temp_corr is None or pd.isna(temp_corr): return False
    if not (PHYS_MIN < float(temp_corr) < PHYS_MAX): return False
    if not (TEMP_MIN < float(temp_corr) < TEMP_MAX): return False
    if (temp_obs is not None) and (not pd.isna(temp_obs)):
        return abs(float(temp_corr) - float(temp_obs)) < OUTLIER_TH_OBS
    else:
        return abs(float(temp_corr) - float(temp_pred)) < OUTLIER_TH

def render_cell_html(
    temp_pred: Optional[float],
    speed_mps: Optional[float],
    dir_deg: Optional[float],
    temp_corr_raw: Optional[float],
    corr_on: bool,
    temp_obs: Optional[float] = None,
) -> str:
    corr_ok = corr_on and correction_effective(temp_pred, temp_corr_raw, temp_obs=temp_obs)
    bg_value = float(temp_corr_raw) if corr_ok else (float(temp_pred) if temp_pred is not None else np.nan)
    bg_color = get_color(bg_value) if not pd.isna(bg_value) else "rgba(220,220,220,0.6)"
    pred_label = f"{float(temp_pred):.1f}°C" if (temp_pred is not None and not pd.isna(temp_pred)) else "NaN"
    pred_html = f"<span class='pred-small'>{pred_label}</span>"

    speed_html, arrow_html = "", ""
    if (speed_mps is not None and not pd.isna(speed_mps)) and (dir_deg is not None and not pd.isna(dir_deg)):
        speed_kt = float(speed_mps) * 1.94384
        speed_html = f"<span style='font-size:12px;color:#444;'>{speed_kt:.1f} kt</span>"
        arrow_html = f"<span style='display:block;line-height:1;margin:0;padding:0;'>{get_arrow_svg(float(dir_deg), float(speed_mps))}</span>"

    corr_html = ""
    if corr_on and (temp_corr_raw is not None) and not pd.isna(temp_corr_raw) and corr_ok:
        corr_html = f"<span style='color:#D32F2F;font-weight:700;font-size:14px;'>{float(temp_corr_raw):.1f}°C</span>"

    content = (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>"
        + pred_html + speed_html + arrow_html + corr_html + "</div>"
    )
    return f"<td style='background:{bg_color}'>{content}</td>"


def build_weekly_table_html(df_period: pd.DataFrame, day_list: List[pd.Timestamp], depths: List[int], corr_on: bool) -> str:
    times = [d.strftime('%m/%d') for d in day_list]
    html = (
        '<div class="calendar-scroll-container"><table class="calendar-table">'
        "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times]) + "</tr></thead><tbody>"
    )

    for depth in depths:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"

        for day in day_list:
            g = df_period[(df_period["date_day"] == day.date()) & (df_period["depth_m"] == depth)]

            if not g.empty:
                series = _pick_series_corr_then_pred(g)
                temp_rep = np.nan
                if series is not None:
                    vv = pd.to_numeric(series, errors="coerce")
                    vv = vv[(vv > PHYS_MIN) & (vv < PHYS_MAX)]
                    if vv.notna().sum() >= 1:
                        temp_rep = float(vv.median())

                # pred表示（小さい文字）は「predの日中央値」固定にする（わかりやすさ優先）
                pred_rep = np.nan
                if "pred_temp" in g.columns:
                    vp = pd.to_numeric(g["pred_temp"], errors="coerce")
                    vp = vp[(vp > PHYS_MIN) & (vp < PHYS_MAX)]
                    if vp.notna().sum() >= 1:
                        pred_rep = float(vp.median())

                # corr表示（赤字）は「corrの日中央値」ただしcorr有効時のみ
                corr_rep = None
                if corr_on and ("corr_temp" in g.columns):
                    vc = pd.to_numeric(g["corr_temp"], errors="coerce")
                    vc = vc[(vc > PHYS_MIN) & (vc < PHYS_MAX)]
                    if vc.notna().sum() >= 1:
                        corr_rep = float(vc.median())

                # obsは「その日の代表（中央値）」
                obs_rep = None
                if "obs_temp" in g.columns:
                    vo = pd.to_numeric(g["obs_temp"], errors="coerce")
                    vo = vo[(vo > PHYS_MIN) & (vo < PHYS_MAX)]
                    if vo.notna().sum() >= 1:
                        obs_rep = float(vo.median())

                # -----------------------------
                # (B) 流れ：
                # -----------------------------
                target_dt = pd.Timestamp(day.date()) + pd.Timedelta(hours=12)
                row = g.assign(_diff=(g["datetime"] - target_dt).abs()).sort_values("_diff").iloc[[0]]

                speed_val = float(row["Speed"].values[0]) if "Speed" in row.columns else np.nan
                dir_val   = float(row["Direction_deg"].values[0]) if "Direction_deg" in row.columns else np.nan

                # -----------------------------
                # セル描画：温度は代表値で渡す
                # - pred側は pred_rep
                # - corr側は corr_rep（Noneなら表示されない）
                # -----------------------------
                html += render_cell_html(
                    temp_pred=pred_rep,
                    speed_mps=speed_val,
                    dir_deg=dir_val,
                    temp_corr_raw=corr_rep,
                    corr_on=corr_on,
                    temp_obs=obs_rep,
                )

            else:
                html += "<td>-</td>"

        html += "</tr>\n"

    html += "</tbody></table></div>"
    return html


def build_daily_table_html(df_day: pd.DataFrame, depths: List[int], corr_on: bool) -> str:
    hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
    times_hr = [t.strftime('%H:%M') for t in hours_list]
    html = (
        '<div class="calendar-scroll-container"><table class="calendar-table">'
        "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times_hr]) + "</tr></thead><tbody>"
    )
    for depth in depths:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for t_obj in hours_list:
            row = df_day[(df_day["datetime"].dt.floor("h") == t_obj) & (df_day["depth_m"] == depth)]
            if not row.empty:
                temp_pred = float(row["pred_temp"].values[0]) if "pred_temp" in row.columns else np.nan
                speed_val = float(row["Speed"].values[0]) if "Speed" in row.columns else np.nan
                dir_val = float(row["Direction_deg"].values[0]) if "Direction_deg" in row.columns else np.nan
                temp_corr = float(row["corr_temp"].values[0]) if "corr_temp" in row.columns else None
                temp_obs = float(row["obs_temp"].values[0]) if ("obs_temp" in row.columns and not pd.isna(row["obs_temp"].values[0])) else None
                html += render_cell_html(temp_pred, speed_val, dir_val, temp_corr, corr_on, temp_obs=temp_obs)
            else:
                html += "<td>-</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>"
    return html

def make_layer_groups(depths: List[int]) -> Dict[str, List[int]]:
    if not depths:
        return {"表層": [], "中層": [], "底層": []}
    d_sorted = sorted(depths); n = len(d_sorted)
    if n <= 3:
        top = d_sorted[:1]
        mid = d_sorted[1:2] if n >= 2 else []
        bot = d_sorted[2:] if n >= 3 else (d_sorted[-1:] if n >= 1 else [])
    elif n in (4, 5):
        top = d_sorted[:2]; mid = d_sorted[2:3]; bot = d_sorted[3:]
    else:
        top = d_sorted[:2]; bot = d_sorted[-2:]
        mid = [d for d in d_sorted if d not in top + bot]
        if len(mid) >= 3:
            c = len(mid) // 2
            mid = mid[c-1:c+1]
    return {"表層": top, "中層": mid, "底層": bot}

def summarize_weekly_for_depth(layer_name: str, target_depth: int, df_period: pd.DataFrame) -> Optional[str]:
    if df_period.empty or "depth_m" not in df_period.columns:
        return None
    g = df_period[df_period["depth_m"] == int(target_depth)].sort_values("datetime")
    if g.empty:
        return None

    series = _pick_series_corr_then_pred(g)
    if series is None:
        return None

    dfz = g.assign(val=pd.to_numeric(series, errors="coerce"))
    dfz = dfz[(dfz["val"] > PHYS_MIN) & (dfz["val"] < PHYS_MAX)].dropna(subset=["val"])
    if dfz.empty:
        return None
    if "date_day" not in dfz.columns:
        dfz["date_day"] = dfz["datetime"].dt.date

    daily = (
        dfz.groupby("date_day", as_index=False)["val"]
        .median()
        .sort_values("date_day")
    )
    temps = daily["val"]
    if temps.empty:
        return None

    rng_th = float(RANGE_STABLE)
    dlt_th = float(DELTA_THRESH)

    t_min, t_max = float(temps.min()), float(temps.max())
    if t_max >= HIGH_TEMP_TH:
        tag = f":red[高水温]（{t_min:.1f}℃～{t_max:.1f}℃）"
        return f"**{layer_name}**： {int(target_depth)}m{tag}"

    weekly_range = t_max - t_min
    if weekly_range < rng_th:
        t_start = float(temps.iloc[0])
        tag = f"安定（{t_start:.1f}℃）"
        return f"**{layer_name}**： {int(target_depth)}m{tag}"

    n = len(temps)
    idx_first = [i for i in [0, 1, 2] if i < n]
    idx_last = [i for i in [4, 5, 6] if i < n]
    first = temps.iloc[idx_first] if idx_first else temps.iloc[:max(1, n // 2)]
    last  = temps.iloc[idx_last]  if idx_last  else temps.iloc[max(1, n // 2):]
    delta = float(last.mean() - first.mean())

    first_mean = float(first.mean()); last_mean = float(last.mean())
    def payload_arrow() -> str: return f"{first_mean:.1f}℃→{last_mean:.1f}℃"
    def payload_range() -> str: return f"{t_min:.1f}–{t_max:.1f}℃"
    def payload() -> str: return payload_arrow() if DISPLAY_MODE == "arrow" else payload_range()

    if delta > +dlt_th:
        tag = f"上昇（{payload()}）"
    elif delta < -dlt_th:
        tag = f"下降（{payload()}）"
    else:
        t_start = float(temps.iloc[0]); t_end = float(temps.iloc[-1])
        end_diff = t_end - t_start
        if abs(end_diff) >= dlt_th:
            tag = f"{'上昇' if end_diff > 0 else '下降'}（{payload()}）"
        else:
            tag = f"安定（{payload()}）"
    return f"**{layer_name}**： {int(target_depth)}m{tag}"

def pick_shallow_mid_deep_min10_from_depths(depths: List[int]) -> List[int]:
    if not depths:
        return []
    xs = sorted(set(int(d) for d in depths))
    n = len(xs)
    if n <= 2:
        return xs
    low_idx = 0
    for i, d in enumerate(xs):
        if d >= 10:
            low_idx = i
            break
    high_idx = n - 1
    mid_idx = (low_idx + high_idx) // 2
    chosen = [xs[low_idx], xs[mid_idx], xs[high_idx]]
    return sorted(set(chosen))

def summarize_weekly_layer_temp(layer_name: str, layer_depths: List[int], df_period: pd.DataFrame) -> Optional[str]:
    if not layer_depths or df_period.empty or "depth_m" not in df_period.columns:
        return None
    valid_depths = set(pd.to_numeric(df_period["depth_m"], errors="coerce").dropna().astype(int))
    depths_in_data = sorted(int(d) for d in layer_depths if int(d) in valid_depths)
    if not depths_in_data:
        return None
    smd = pick_shallow_mid_deep_min10_from_depths(depths_in_data)
    if not smd:
        return None
    if layer_name == "表層":
        target_depth = smd[0]
    elif layer_name == "中層":
        target_depth = smd[min(1, len(smd)-1)]
    else:
        target_depth = smd[-1]
    return summarize_weekly_for_depth(layer_name, target_depth, df_period)

def dir_to_8pt_jp(deg: float) -> str:
    if pd.isna(deg): return ""
    dirs = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
    idx = int(((float(deg) + 22.5) % 360) // 45)
    return dirs[idx]

def speed_class_from_mps(v_mps: Optional[float]) -> str:
    if v_mps is None or pd.isna(v_mps): return ""
    kt = float(v_mps) * 1.94384
    if kt >= 1.5: return "速"
    if kt >= 0.5: return "中"
    return "穏"

def summarize_daily_layer_flow(
    layer_name: str,
    layer_depths: List[int],
    df_day: pd.DataFrame,
    use_short_labels: bool = True,
    merge_same_segments: bool = False
) -> Optional[str]:
    if not layer_depths: return None
    DAY_BINS = [("朝", 4, 6), ("昼", 11, 13), ("夕", 16, 18)]
    order = {"朝": 0, "昼": 1, "夕": 2}
    rows: List[Tuple[str, str, str]] = []
    for label, h0, h1 in DAY_BINS:
        g = df_day[(df_day["depth_m"].isin(layer_depths)) & (df_day["datetime"].dt.hour.between(h0, h1))]
        if g.empty: continue
        U_mean = g["U"].mean() if "U" in g.columns else np.nan
        V_mean = g["V"].mean() if "V" in g.columns else np.nan
        if pd.notna(U_mean) and pd.notna(V_mean):
            speed_mean = float(np.sqrt(U_mean**2 + V_mean**2))
            dir_deg_mean = (np.degrees(np.arctan2(U_mean, V_mean)) + 360.0) % 360.0
        else:
            D = g["Direction_deg"].dropna() if "Direction_deg" in g.columns else pd.Series(dtype=float)
            if D.empty: continue
            rad = np.deg2rad(D.values)
            C = np.cos(rad).mean(); S = np.sin(rad).mean()
            dir_deg_mean = (np.degrees(np.arctan2(S, C)) + 360.0) % 360.0
            speed_mean = g["Speed"].mean() if "Speed" in g.columns else np.nan
        d_txt = dir_to_8pt_jp(dir_deg_mean) if pd.notna(dir_deg_mean) else ""
        v_cls = speed_class_from_mps(speed_mean) if pd.notna(speed_mean) else ""
        if use_short_labels and v_cls:
            v_map = {"穏やか": "穏", "中程度": "中", "速い": "速"}
            v_cls = v_map.get(v_cls, v_cls)
        if d_txt or v_cls:
            rows.append((label, d_txt, v_cls))
    if not rows: return None

    segments: List[str] = []
    if merge_same_segments:
        bucket: Dict[Tuple[str, str], List[str]] = {}
        for lbl, d, v in rows: bucket.setdefault((d, v), []).append(lbl)
        for (d, v), lbls in bucket.items():
            lbls_sorted = sorted(lbls, key=lambda x: order.get(x, 99))
            inner = "・".join([x for x in [d, v] if x])
            segments.append(f"{'・'.join(lbls_sorted)}（{inner}）")
    else:
        rows_sorted = sorted(rows, key=lambda r: order.get(r[0], 99))
        for lbl, d, v in rows_sorted:
            inner = "・".join([x for x in [d, v] if x])
            segments.append(f"{lbl}（{inner}）")
    return f"**{layer_name}**： " + "／".join(segments)

# =========================================
# メインUI
# =========================================
inject_compact_css()

try:
    view_mode = st.segmented_control(
        "",  # ラベル非表示
        options=["ガイダンス", "水温図", "CMEM"],
        default="ガイダンス"
    )
except Exception:
    view_mode = st.radio(
        "", ["ガイダンス", "水温図", "CMEM"],
        index=0, horizontal=True, label_visibility="collapsed"
    )

# pred ファイル選択（ラベル非表示）
# - CMEM だけ閲覧したいケースに対応するため、CMEM モードでは pred を必須にしない
selected_file = None
pred_path = corr_path = obs_path = ""
fp_pred = fp_corr = fp_obs = ""

if view_mode != "CMEM":
    pred_folder = pjoin(BASE_DIR, PRED_DIR)
    if not os.path.exists(pred_folder):
        st.error(f"フォルダが見つかりません: {pred_folder}")
        st.stop()

    pred_files = [f for f in os.listdir(pred_folder) if f.endswith(".csv")]
    if not pred_files:
        st.warning("pred に CSV がありません")
        st.stop()

    selected_file = st.selectbox(
        "", sorted(pred_files), key="sel_pred_file", label_visibility="collapsed"
    )

    pred_path = pjoin(BASE_DIR, PRED_DIR, selected_file)
    corr_name, ext = os.path.splitext(selected_file)
    corr_path = pjoin(BASE_DIR, CORR_DIR, f"{corr_name}_corr{ext}")
    obs_path = pjoin(BASE_DIR, OBS_DIR, selected_file)

    # 指紋（キャッシュキー）
    fp_pred = file_fingerprint(pred_path)
    fp_corr = file_fingerprint(corr_path)
    fp_obs  = file_fingerprint(obs_path)

# ガイダンス
if view_mode == "ガイダンス":
    df_pred = load_pred(selected_file, fp_pred)
    df_corr = load_corr_for(selected_file, fp_corr)
    df_obs  = load_obs_for(selected_file,  fp_obs)
    corr_available = not df_corr.empty

    if df_pred.empty:
        st.warning("予測データが読み込めませんでした")
        st.stop()

    # ここで今日（JST）を取得（毎セッション初期評価時に自動で今日へ）
    today_jst = pd.Timestamp.now(tz="Asia/Tokyo").date()

    latest_day = df_pred["date_day"].max()
    available_days = sorted(df_pred["date_day"].unique())
    min_day = min(available_days) if available_days else latest_day
    max_day = max(available_days) if available_days else latest_day

    try:
        cal_choice = st.segmented_control(
            "", options=["週間表示", "選択日"], default="週間表示", key="cal_choice"
        )
    except Exception:
        cal_choice = st.radio(
            "", ["週間表示", "選択日"],
            index=0, horizontal=True, key="cal_choice_radio", label_visibility="collapsed"
        )

    if cal_choice == "週間表示":
        base_day_week = min(max(today_jst, min_day), max_day)

        selected_day = st.date_input(
            "", value=base_day_week, min_value=min_day, max_value=max_day,
            key="week_base_day", label_visibility="collapsed"
        )

        if WEEK_WINDOW_FORWARD:
            start_day = pd.Timestamp(selected_day)
            end_day   = start_day + pd.Timedelta(days=7)  
        else:
            end_day   = pd.Timestamp(selected_day)
            start_day = end_day - pd.Timedelta(days=7)    

        day_list = list(pd.date_range(start_day, end_day, freq="D"))

        df_period = df_pred[df_pred["date_day"].isin([d.date() for d in day_list])].copy()
        if corr_available:
            df_corr_period = df_corr[df_corr["date_day"].isin([d.date() for d in day_list])].copy()
            df_period = add_corr(df_period, df_corr_period)

        if not df_obs.empty and not df_period.empty:
            df_obs_week = df_obs[df_obs["date_day"].between(day_list[0].date(), day_list[-1].date())].copy()
            tol_obs = pd.Timedelta(minutes=OBS_MATCH_TOL_MIN)
            left = df_period.sort_values(["depth_m", "datetime"]).copy()
            right = df_obs_week.sort_values(["depth_m", "datetime"])[["datetime", "depth_m", "obs_temp"]].copy()
            merged = safe_merge_asof_by_depth_keep_left(left, right, tolerance=tol_obs, right_value_cols=["obs_temp"], suffixes=("", ""))
            if "obs_temp" in merged.columns:
                df_period = merged

        depths_all = sorted([int(d) for d in df_pred["depth_m"].dropna().unique()])
        st.markdown(f"**{start_day:%m/%d}～{end_day:%m/%d}の推移**")

        reps = pick_shallow_mid_deep_min10_from_depths(depths_all)
        mapping = []
        if len(reps) >= 1: mapping.append(("表層", reps[0]))
        if len(reps) >= 2: mapping.append(("中層", reps[min(1, len(reps)-1)]))
        if len(reps) >= 3: mapping.append(("底層", reps[-1]))

        any_line = False
        for lname, depth_sel in mapping:
            line = summarize_weekly_for_depth(lname, depth_sel, df_period)
            if line:
                any_line = True
                st.markdown(line)
        if not any_line:
            st.caption("（特筆すべき変化はありません）")

        table_html = build_weekly_table_html(df_period, day_list, depths_all, corr_on=corr_available)
        styles = get_calendar_css(65)
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        st_html(full_html, height=650, scrolling=True)

    else:  
        if "date_day" in df_pred.columns:
            _days = sorted(df_pred["date_day"].dropna().unique())
            if _days:
                _min_day = min(_days)
                _max_day = max(_days)
            else:
                _min_day = latest_day
                _max_day = latest_day
        else:
            _min_day = latest_day
            _max_day = latest_day

        base_day_day = min(max(pd.Timestamp.now(tz="Asia/Tokyo").date(), _min_day), _max_day)

        selected_day = st.date_input(
            "", value=base_day_day, min_value=_min_day, max_value=_max_day,
            key="day_sel", label_visibility="collapsed"
        )

        df_day = df_pred[df_pred["date_day"] == selected_day].copy()
        if corr_available:
            df_corr_sel = df_corr[df_corr["date_day"] == selected_day].copy()
            df_day = add_corr(df_day, df_corr_sel)

        if not df_obs.empty and not df_day.empty:
            df_obs_sel = df_obs[df_obs["date_day"] == selected_day].copy()
            tol_obs = pd.Timedelta(minutes=OBS_MATCH_TOL_MIN)
            left = df_day.sort_values(["depth_m", "datetime"]).copy()
            right = df_obs_sel.sort_values(["depth_m", "datetime"])[["datetime", "depth_m", "obs_temp"]].copy()
            merged = safe_merge_asof_by_depth_keep_left(left, right, tolerance=tol_obs, right_value_cols=["obs_temp"], suffixes=("", ""))
            if "obs_temp" in merged.columns:
                df_day = merged

        depths_all = sorted([int(d) for d in df_pred["depth_m"].dropna().unique()])
        st.markdown("**朝(4～6時)、昼(11～13時)、夕(16～18時)**")
        layers = make_layer_groups(depths_all)

        any_line = False
        for lname, ldepths in layers.items():
            line = summarize_daily_layer_flow(lname, ldepths, df_day)
            if line:
                any_line = True
                st.markdown(line)
        if not any_line:
            st.caption("（特筆すべき変化はありません）")

        table_html = build_daily_table_html(df_day, depths_all, corr_on=corr_available)
        styles = get_calendar_css(65)
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        st_html(full_html, height=650, scrolling=True)

elif view_mode == "CMEM":
    sites = list_cmem_sites()
    if not sites:
        st.warning("data/cmem/thetao と data/cmem/chl にサイトCSVがありません")
        st.stop()

    if selected_file:
        site_guess = os.path.splitext(selected_file)[0]
        if site_guess not in sites:
            st.warning(f"CMEMに site '{site_guess}' がありません（data/cmem/thetao, chl を確認）")
            st.stop()
        sel_site = site_guess
    else:
        sel_site = st.selectbox("", sites, key="cmem_site", label_visibility="collapsed")

    path_t = pjoin(BASE_DIR, CMEM_DIR, CMEM_THETAO_DIR, f"thetao_{sel_site}.csv")
    path_c = pjoin(BASE_DIR, CMEM_DIR, CMEM_CHL_DIR, f"chl_{sel_site}.csv")
    fp_t = file_fingerprint(path_t)
    fp_c = file_fingerprint(path_c)
    df_t = load_cmem_thetao(sel_site, fp_t)
    df_c = load_cmem_chl(sel_site, fp_c)

    if df_t.empty and df_c.empty:
        st.warning("CMEMデータが読み込めませんでした")
        st.stop()

    show_thetao = (not df_t.empty)
    show_chl = (not df_c.empty)

    depths_t = sorted([int(d) for d in df_t['depth_m'].dropna().astype(int).unique()]) if (show_thetao and (not df_t.empty)) else []
    depths_c = sorted([int(d) for d in df_c['depth_m'].dropna().astype(int).unique()]) if (show_chl and (not df_c.empty)) else []
    depths_all = sorted(set(depths_t + depths_c))
    if not depths_all:
        st.warning("深度情報がありません")
        st.stop()
    selected_depths = depths_all
    depths_sorted = depths_all

    dt_col = 'datetime'
    depths_int = [int(d) for d in selected_depths]
    df_t2_raw = df_t[df_t['depth_m'].isin(depths_int)].copy() if (show_thetao and not df_t.empty) else pd.DataFrame()
    df_c2_raw = df_c[df_c['depth_m'].isin(depths_int)].copy() if (show_chl and not df_c.empty) else pd.DataFrame()

    def _cmem_dt_minmax(df_a: pd.DataFrame, df_b: pd.DataFrame):
        s = pd.Series(dtype='datetime64[ns]')
        if not df_a.empty: s = pd.concat([s, pd.to_datetime(df_a[dt_col], errors='coerce')])
        if not df_b.empty: s = pd.concat([s, pd.to_datetime(df_b[dt_col], errors='coerce')])
        s = s.dropna()
        if len(s) == 0: return None, None
        return s.min(), s.max()

    try:
        cmem_period = st.segmented_control("", options=["日別", "月別"], default="日別", key="cmem_period")
    except Exception:
        cmem_period = st.radio("", ["日別", "月別"], index=0, horizontal=True, key="cmem_period_radio", label_visibility="collapsed")

    tab_cmem_ts, tab_cmem_md = st.tabs(["時系列", "同月日比較"])

    def _render_cmem(cmem_view: str, df_t2: pd.DataFrame, df_c2: pd.DataFrame):
        def _available_years():
            s = pd.Series(dtype='datetime64[ns]')
            if not df_t2.empty:
                s = pd.concat([s, pd.to_datetime(df_t2[dt_col], errors='coerce')])
            if not df_c2.empty:
                s = pd.concat([s, pd.to_datetime(df_c2[dt_col], errors='coerce')])
            s = s.dropna()
            return sorted(s.dt.year.unique().tolist()) if len(s) else []

        years_all = _available_years()

        base_year = None
        comp_years = []
        if cmem_view == "同月日比較":
            if not years_all:
                st.warning("年情報がありません")
                st.stop()

            years_sorted = sorted([int(y) for y in years_all])
            base_year = st.selectbox("", years_sorted, index=len(years_sorted)-1, key="cmem_base_year", label_visibility="collapsed")
            cand = [y for y in years_sorted if y != int(base_year)]
            default_comp = cand[-2:] if len(cand) >= 2 else cand
            comp_years = st.multiselect("", cand, default=default_comp, key="cmem_comp_years", label_visibility="collapsed")
            if not comp_years:
                st.info("比較する年を選択してください")
                st.stop()

        base_colors = px.colors.qualitative.Dark24

        if cmem_view == "時系列":
            def _prep_grid(df_in: pd.DataFrame, value_col: str):
                if df_in.empty:
                    return None, None, None
                dfw = df_in.copy()
                dts = pd.to_datetime(dfw[dt_col], errors='coerce')
                if cmem_period == "月別":
                    dfw['t'] = dts.dt.to_period('M').dt.to_timestamp()
                else:
                    dfw['t'] = dts.dt.floor('D')
                dfw = dfw.dropna(subset=['depth_m','t', value_col]).copy()
                if dfw.empty:
                    return None, None, None
                dfw['depth_m'] = pd.to_numeric(dfw['depth_m'], errors='coerce').round(0).astype('Int64')
                dfw[value_col] = pd.to_numeric(dfw[value_col], errors='coerce')
                dfw = dfw.dropna(subset=['depth_m','t', value_col]).copy()
                dfw = dfw.groupby(['depth_m','t'], as_index=False)[value_col].mean()

                depths = sorted([int(d) for d in dfw['depth_m'].dropna().astype(int).unique().tolist()])
                times = sorted(pd.to_datetime(dfw['t']).dropna().unique().tolist())
                if (not depths) or (not times):
                    return None, None, None

                piv = dfw.pivot(index='depth_m', columns='t', values=value_col)
                piv = piv.reindex(index=depths, columns=times)
                z = piv.values
                return times, depths, z

            rows = 0
            titles = []
            grids = []
            if show_thetao and (not df_t2.empty):
                x, y, z = _prep_grid(df_t2, 'thetao')
                if x is not None:
                    rows += 1
                    titles.append("thetao（水温）")
                    grids.append(('thetao', x, y, z))
            if show_chl and (not df_c2.empty):
                df_c2 = df_c2.copy()
                df_c2['chl_log'] = np.log10(np.maximum(pd.to_numeric(df_c2['chl'], errors='coerce'), 0.01))
                x, y, z = _prep_grid(df_c2, 'chl_log')
                if x is not None:
                    rows += 1
                    titles.append("log10(chl)")
                    grids.append(('chl_log', x, y, z))

            if rows == 0:
                st.warning("CMEMデータが表示できません（空）")
                st.stop()

            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.14,
                subplot_titles=titles
            )

            def _cbar_y(r: int) -> float:
                return 1.0 - (r - 0.5) / rows

            for r, (vname, x, y, z) in enumerate(grids, start=1):
                if vname == 'thetao':
                    colorscale = 'RdBu_r'
                    colorbar_title = '℃'
                else:
                    colorscale = 'Viridis'
                    colorbar_title = 'log10(chl)'

                tr = go.Contour(
                    x=x, y=y, z=z,
                    colorscale=colorscale,
                    contours=dict(coloring='heatmap', showlines=False),
                    ncontours=20,
                    colorbar=dict(title=colorbar_title, x=1.02, y=_cbar_y(r), yanchor='middle', len=0.75/rows),
                    hovertemplate="%{x}<br>Depth: %{y} m<br>Value: %{z:.4g}<extra></extra>"
                )
                fig.add_trace(tr, row=r, col=1)
                fig.update_yaxes(autorange='reversed', title_text="水深 (m)", row=r, col=1)

            title_suffix = "（時系列・月平均）" if cmem_period == "月別" else "（時系列・日別）"
            fig.update_layout(
                title={"text": f"CMEM {sel_site}{title_suffix}", "y": 0.98, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
                margin=dict(l=10, r=120, t=70, b=10),
                height=260 + 280 * rows,
                template="plotly_white",
            )
            fig.update_xaxes(title_text="日付" if cmem_period == "日別" else "月", row=rows, col=1)
            st.plotly_chart(fig, use_container_width=True)

        else:
            base_y = int(base_year)
            comp_sorted = sorted([int(y) for y in comp_years if int(y) != base_y])
            if not comp_sorted:
                st.info("比較する年を選択してください")
                st.stop()

            if cmem_period == "月別":
                m_start, m_end = st.slider(
                    "", min_value=1, max_value=12, value=(1, 12),
                    key="cmem_month_window", label_visibility="collapsed"
                )
                months = list(range(int(m_start), int(m_end) + 1)) 
                month_order = {m: i for i, m in enumerate(months)}
                xname = "x_idx"
            else:
                m_start, m_end = st.slider(
                    "", min_value=1, max_value=12, value=(1, 12),
                    key="cmem_md_month_window", label_visibility="collapsed"
                )
                start_dt = pd.Timestamp(year=2000, month=int(m_start), day=1)
                end_dt   = (pd.Timestamp(year=2000, month=int(m_end), day=1) + pd.offsets.MonthEnd(0))
                md_list  = [d.strftime("%m-%d") for d in pd.date_range(start_dt, end_dt, freq="D")]
                md_order = {m: k for k, m in enumerate(md_list)}
                xname = "x_idx"

            def prep_thetao(df):
                if df.empty:
                    return pd.DataFrame()
                dts = pd.to_datetime(df[dt_col])
                df = df.assign(y=dts.dt.year)

                if cmem_period == "月別":
                    df = df.assign(
                        m=dts.dt.month,
                        x_idx=dts.dt.month.map(month_order),
                        x_label=dts.dt.month.apply(lambda v: f"{int(v)}月")
                    )
                    df = df[(df["m"] >= m_start) & (df["m"] <= m_end)]
                else:
                    df = df.assign(md=dts.dt.strftime("%m-%d"))
                    df = df[df["md"].isin(md_order)]
                    df = df.assign(x_idx=df["md"].map(md_order), x_label=df["md"])
                return df.groupby(
                    ["depth_m", "y", xname, "x_label"],
                    as_index=False
                )["thetao"].mean()

            def prep_chl(df):
                if df.empty:
                    return pd.DataFrame()
                dts = pd.to_datetime(df[dt_col])
                df = df.assign(
                    y=dts.dt.year,
                    chl_log=np.log10(np.maximum(pd.to_numeric(df["chl"], errors="coerce"), 0.01))
                )

                if cmem_period == "月別":
                    df = df.assign(
                        m=dts.dt.month,
                        x_idx=dts.dt.month.map(month_order),
                        x_label=dts.dt.month.apply(lambda v: f"{int(v)}月")
                    )
                    df = df[(df["m"] >= m_start) & (df["m"] <= m_end)]
                else:
                    df = df.assign(md=dts.dt.strftime("%m-%d"))
                    df = df[df["md"].isin(md_order)]
                    df = df.assign(x_idx=df["md"].map(md_order), x_label=df["md"])
                return df.groupby(
                    ["depth_m", "y", xname, "x_label"],
                    as_index=False
                )["chl_log"].mean()

            df_tg = prep_thetao(df_t2) if show_thetao else pd.DataFrame()
            df_cg = prep_chl(df_c2)    if show_chl else pd.DataFrame()

            def diff_base(df, valcol):
                if df.empty:
                    return pd.DataFrame()
                base = df[df["y"] == base_y][["depth_m", xname, "x_label", valcol]].rename(columns={valcol: "base"})
                cmp = df[df["y"].isin(comp_sorted)][["depth_m", xname, "x_label", valcol]]
                cmp_mean = cmp.groupby(["depth_m", xname, "x_label"], as_index=False)[valcol].mean()
                cmp_mean = cmp_mean.rename(columns={valcol: "cmp"})
                out = pd.merge(base, cmp_mean, on=["depth_m", xname, "x_label"])
                out["diff"] = out["base"] - out["cmp"]   
                return out

            df_tdiff = diff_base(df_tg, "thetao")
            df_cdiff = diff_base(df_cg, "chl_log")

            if cmem_period == "月別":
                x_grid = list(range(len(months)))
                x_labels = [f"{m}月" for m in months]
                tickvals = x_grid
                ticktext = x_labels

            else:
                x_grid = list(range(len(md_list)))
                x_labels = md_list[:]
                dt_tmp = pd.to_datetime([f"2000-{s}" for s in md_list], errors="coerce")
                tickvals, ticktext = [], []
                for i, d in enumerate(dt_tmp):
                    if pd.notna(d) and d.day == 1:
                        tickvals.append(i)
                        ticktext.append(d.strftime("%m/%d"))
                if len(tickvals) == 0:
                    step = max(1, len(md_list) // 12)
                    tickvals = list(range(0, len(md_list), step))
                    ticktext = [md_list[i] for i in tickvals]
        
            def _pivot_z(df_diff: pd.DataFrame) -> np.ndarray:
                """depths_sorted × x_grid の z 行列を作る（欠損は NaN）。"""
                if df_diff.empty:
                    return np.full((len(depths_sorted), len(x_grid)), np.nan)
                pv = (
                    df_diff.pivot_table(index="depth_m", columns=xname, values="diff", aggfunc="mean")
                    .reindex(index=depths_sorted, columns=x_grid)
                )
                return pv.values
        
            def _sym_zrange(z: np.ndarray, fallback: float = 1.0) -> float:
                """0中心の対称レンジ用 maxabs を返す。"""
                if z.size == 0:
                    return fallback
                m = np.nanmax(np.abs(z))
                if (not np.isfinite(m)) or (m <= 0):
                    return fallback
                return float(m)
        
            def _custom_xlabels_2d() -> np.ndarray:
                return np.tile(np.array(x_labels, dtype=object), (len(depths_sorted), 1))
        
            show_t = (show_thetao and (not df_tdiff.empty))
            show_c = (show_chl and (not df_cdiff.empty))
            if (not show_t) and (not show_c):
                st.warning("差分を描画できるデータがありません（基準年・比較年・期間・深度を確認）")
                st.stop()
        
            nrows = (1 if (show_t ^ show_c) else 2)
            titles = []
            if show_t:
                titles.append("thetao（水温）差：基準 − 比較（平均）")
            if show_c:
                titles.append("chl（log10）差：基準 − 比較（平均）")
        
            if nrows == 1:
                fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=titles)
            else:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14, subplot_titles=titles)
        
            row_i = 1
        
            if show_t:
                zt = _pivot_z(df_tdiff)
                maxabs_t = _sym_zrange(zt, fallback=0.5)
                fig.add_trace(
                    go.Contour(
                        x=x_grid, y=depths_sorted, z=zt,
                        colorscale="RdBu_r", zmin=-maxabs_t, zmax=maxabs_t,
                        contours=dict(coloring="heatmap"), connectgaps=False,
                        colorbar=dict(title="℃", x=1.02, y=(0.78 if nrows == 2 else 0.50), len=(0.42 if nrows == 2 else 0.85)),
                        customdata=_custom_xlabels_2d(),
                        hovertemplate="時点:%{customdata}<br>水深:%{y} m<br>差:%{z:.2f} ℃<extra></extra>"
                    ),
                    row=row_i, col=1
                )

                fig.update_yaxes(autorange="reversed", title_text="水深 (m)", row=row_i, col=1)
                row_i += 1
        
            if show_c:
                zc = _pivot_z(df_cdiff)
                maxabs_c = _sym_zrange(zc, fallback=0.3)
                fig.add_trace(
                    go.Contour(
                        x=x_grid, y=depths_sorted, z=zc,
                        colorscale="RdBu_r", zmin=-maxabs_c, zmax=maxabs_c,
                        contours=dict(coloring="heatmap"), connectgaps=False,
                        colorbar=dict(title="log10", x=1.02, y=(0.22 if nrows == 2 else 0.50), len=(0.42 if nrows == 2 else 0.85)),
                        customdata=_custom_xlabels_2d(),
                        hovertemplate="時点:%{customdata}<br>水深:%{y} m<br>差:%{z:.2f} (log10)<extra></extra>"
                    ),
                    row=row_i, col=1
                )

                fig.update_yaxes(autorange="reversed", title_text="水深 (m)", row=row_i, col=1)
       
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
            fig.update_xaxes(title_text=("月" if cmem_period == "月別" else "月日"), row=nrows, col=1)
            fig.update_layout(
                title=f"CMEM {sel_site} 同月比較差分（{base_y} − 平均[{','.join(map(str, comp_sorted))}]）",
                height=(520 if nrows == 1 else 760),
                margin=dict(l=60, r=110, t=60, b=50),
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab_cmem_ts:
        dt_min, dt_max = _cmem_dt_minmax(df_t2_raw, df_c2_raw)
        if dt_min is None or dt_max is None:
            st.warning("日時情報がありません")
            st.stop()
        d0 = pd.to_datetime(dt_min).date()
        d1 = pd.to_datetime(dt_max).date()
        try:
            _dflt_start = (pd.Timestamp(d1) - pd.DateOffset(years=2)).date()
        except Exception:
            _dflt_start = (pd.Timestamp(d1) - pd.Timedelta(days=365*2)).date()
        _dflt_start = max(d0, _dflt_start)
        d_start, d_end = st.slider("", min_value=d0, max_value=d1, value=(_dflt_start, d1), key="cmem_dt_range_ts_slider", label_visibility="collapsed")
        t_start = pd.Timestamp(d_start)
        t_end = pd.Timestamp(d_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_t2_ts = df_t2_raw.copy()
        df_c2_ts = df_c2_raw.copy()
        if not df_t2_ts.empty:
            _dt = pd.to_datetime(df_t2_ts[dt_col], errors='coerce')
            df_t2_ts = df_t2_ts[(_dt >= t_start) & (_dt <= t_end)].copy()
        if not df_c2_ts.empty:
            _dt = pd.to_datetime(df_c2_ts[dt_col], errors='coerce')
            df_c2_ts = df_c2_ts[(_dt >= t_start) & (_dt <= t_end)].copy()
        _render_cmem("時系列", df_t2_ts, df_c2_ts)
    with tab_cmem_md:
        _render_cmem("同月日比較", df_t2_raw, df_c2_raw)

elif view_mode == "水温図":
    df_pred = load_pred(selected_file, fp_pred)
    df_corr = load_corr_for(selected_file, fp_corr)
    df_obs  = load_obs_for(selected_file,  fp_obs)
    corr_available = not df_corr.empty

    if df_pred.empty:
        st.warning("予測データが読み込めませんでした")
        st.stop()

    latest_dt = df_pred["datetime"].max()
    available_days = sorted(df_pred["date_day"].unique()) if "date_day" in df_pred.columns else []
    if available_days:
        min_day = min(available_days); max_day = max(available_days)
    else:
        min_day = latest_dt.date(); max_day = latest_dt.date()

    try:
        graph_style = st.segmented_control("", options=["コンター", "折れ線"], default="コンター", key="graph_style")
    except Exception:
        graph_style = st.radio("", ["コンター", "折れ線"], index=0, horizontal=True, key="graph_style_radio", label_visibility="collapsed")
    start_default = max(min_day, max_day - pd.Timedelta(days=10))
    start_day, end_day = st.slider(
        "", min_value=min_day, max_value=max_day, value=(start_default, max_day),
        key="graph_period_slider", label_visibility="collapsed"
    )
    title_suffix = f"（{start_day:%Y-%m-%d}〜{end_day:%Y-%m-%d}・時間別）"

    contour_agg = st.session_state.get("graph_contour_agg", "日平均")
    if graph_style == "コンター":
        try:
            contour_agg = st.segmented_control("", options=["1時間", "日平均"], default="1時間", key="graph_contour_agg")
        except Exception:
            contour_agg = st.radio("", ["1時間", "日平均"], index=1, horizontal=True, key="graph_contour_agg_radio", label_visibility="collapsed")

    snap_freq = "1h" if contour_agg == "1時間" else "1D"

    df_period = df_pred[(df_pred["date_day"] >= start_day) & (df_pred["date_day"] <= end_day)].copy()
    df_period = df_period.sort_values("datetime")
    if "pred_temp" in df_period.columns and not df_period.empty:
        df_period = (
            df_period.groupby(["depth_m", "datetime"], as_index=False).agg({"pred_temp": "median"})
        )
    if not df_period.empty:
        df_period = (
            df_period.sort_values("datetime")
            .groupby("depth_m", group_keys=False)
            .apply(lambda g: (
                g.drop(columns=["depth_m"], errors="ignore").set_index("datetime")
                .resample("1h").median(numeric_only=True).interpolate(method="time", limit=2).reset_index()
                .assign(depth_m=int(g.name) if g.name is not None else pd.NA)
            ))
        )
    if "depth_m" in df_period.columns:
        df_period["depth_m"] = pd.to_numeric(df_period["depth_m"], errors="coerce").round(0).astype("Int64")

    merged_for_points = pd.DataFrame(columns=["datetime", "depth_m", "obs_temp"])
    if not df_obs.empty and not df_period.empty:
        df_obs_period = df_obs[(df_obs["date_day"] >= start_day) & (df_obs["date_day"] <= end_day)].copy()
        if not df_obs_period.empty:
            tol = pd.Timedelta(minutes=CORR_MATCH_TOL_MIN)
            left = df_period.sort_values(["depth_m","datetime"]).copy()
            right = df_obs_period.sort_values(["depth_m","datetime"])[["datetime","depth_m","obs_temp"]].copy()
            merged_for_points = safe_merge_asof_by_depth_keep_left(
                left=left, right=right, tolerance=tol, right_value_cols=["obs_temp"], suffixes=("","")
            )

    df_corr_period = pd.DataFrame()
    if corr_available:
        df_corr_period = df_corr[(df_corr["date_day"] >= start_day) & (df_corr["date_day"] <= end_day)].copy()
        if not df_corr_period.empty:
            use_cols = ["corr_temp"]
            if "corr_low" in df_corr_period.columns: use_cols.append("corr_low")
            if "corr_high" in df_corr_period.columns: use_cols.append("corr_high")
            df_corr_period = (
                df_corr_period.sort_values("datetime")
                .groupby("depth_m", group_keys=False)
                .apply(lambda g: (
                    g.drop(columns=["depth_m"], errors="ignore")
                    .set_index("datetime")[use_cols]
                    .resample("1h").median().dropna(how="all").reset_index()
                    .assign(depth_m=int(g.name) if g.name is not None else pd.NA)
                ))
            )

    if graph_style == "折れ線":
            fig = go.Figure()
            base_colors = px.colors.qualitative.Dark24
      
            depths_pred_all = sorted(set(df_period["depth_m"].dropna().astype(int).tolist())) if not df_period.empty else []
            depths_with_corr = set()
            if not df_corr_period.empty and "depth_m" in df_corr_period.columns:
                depths_with_corr = set(pd.to_numeric(df_corr_period["depth_m"], errors="coerce").dropna().astype(int).unique())
       
            depths_with_obs = set()
            if ("depth_m" in merged_for_points.columns) and ("obs_temp" in merged_for_points.columns):
                tmp_obs = merged_for_points.dropna(subset=["obs_temp"])
                if not tmp_obs.empty:
                    depths_with_obs = set(pd.to_numeric(tmp_obs["depth_m"], errors="coerce").dropna().astype(int).unique())
      

            both_corr_obs = sorted(depths_with_corr.intersection(depths_with_obs))
        
            def pick_shallow_mid_deep_min10(cands: List[int], k: int = 3) -> List[int]:
                if not cands:
                    return []
                xs = sorted(set(int(d) for d in cands))
                n = len(xs)
                if n <= 2:
                    return xs[:k]
                low_idx = 0
                for i, d in enumerate(xs):
                    if d >= 10:
                        low_idx = i
                        break
                high_idx = n - 1
                mid_idx = (low_idx + high_idx) // 2
                idxs = [low_idx, mid_idx, high_idx]
                chosen = [xs[i] for i in sorted(set(idxs))]
                if len(chosen) < k:
                    center = xs[mid_idx]
                    rest = [d for d in xs if d not in chosen]
                    rest_sorted = sorted(rest, key=lambda d: (abs(d - center), d))
                    chosen.extend(rest_sorted[:k - len(chosen)])

                return chosen[:k]

        
            if len(both_corr_obs) >= 3:
                default_depths = pick_shallow_mid_deep_min10(both_corr_obs, k=3)

            elif len(depths_with_corr) >= 3:
                default_depths = pick_shallow_mid_deep_min10(sorted(depths_with_corr), k=3)
            else:
                default_depths = pick_shallow_mid_deep_min10(depths_pred_all, k=3)
            if not default_depths:
                default_depths = depths_pred_all[: min(3, len(depths_pred_all))]
       
            selected_depths = st.multiselect(
                "", depths_pred_all, default=default_depths, key="graph_depths", label_visibility="collapsed"
            )
        
            def emphasize_color(hex_color: str) -> str:
                try:
                    rr = int(hex_color[1:3], 16); gg = int(hex_color[3:5], 16); bb = int(hex_color[5:7], 16)
                    rr = min(255, rr + 25); gg = min(255, gg + 25); bb = min(255, bb + 25)
                    return f"#{rr:02x}{gg:02x}{bb:02x}"
                except Exception:
                    return hex_color
        
            for i, d in enumerate(selected_depths):
                base_col = base_colors[i % len(base_colors)]
                corr_col = emphasize_color(base_col)
                lg = f"depth{int(d)}"
        
                g_pred = df_period[df_period["depth_m"] == d]
                g_corr = df_corr_period[df_corr_period["depth_m"] == d] if not df_corr_period.empty else pd.DataFrame()
                g_obs = merged_for_points[merged_for_points["depth_m"] == d] if ("depth_m" in merged_for_points.columns) else pd.DataFrame()
       
                if not g_corr.empty:
                    if ("corr_low" in g_corr.columns) and ("corr_high" in g_corr.columns):
                        fig.add_trace(go.Scatter(
                            x=g_corr["datetime"], y=g_corr["corr_low"].clip(lower=TEMP_MIN, upper=TEMP_MAX),
                            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", name=f"{d}m 帯(下)"
                        ))
                        fig.add_trace(go.Scatter(
                            x=g_corr["datetime"], y=g_corr["corr_high"].clip(lower=TEMP_MIN, upper=TEMP_MAX),
                            mode="lines", line=dict(width=0),
                            fill='tonexty', fillcolor=to_rgba(corr_col, 0.18),
                            name=f"{d}m 信頼帯", legendgroup=lg, showlegend=False, hoverinfo="skip"
                        ))

                    y_corr = g_corr["corr_temp"].clip(lower=TEMP_MIN, upper=TEMP_MAX)
                    fig.add_trace(go.Scatter(
                        x=g_corr["datetime"], y=y_corr, mode="lines",
                        name=f"{d}m 補正", legendgroup=lg, showlegend=True,
                        line=dict(color=corr_col, width=3.0), opacity=1.0,
                        hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>補正水温: %{y:.2f} °C<extra></extra>"
                    ))

                    if not g_pred.empty:
                        y_pred = g_pred["pred_temp"].astype(float).clip(lower=TEMP_MIN, upper=TEMP_MAX)
                        fig.add_trace(go.Scatter(
                            x=g_pred["datetime"], y=y_pred, mode="lines",
                            name=f"{d}m 予測", legendgroup=lg, showlegend=False,
                            line=dict(color=base_col, width=1.2, dash="dot"), opacity=0.35,
                            hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>予測水温: %{y:.2f} °C<extra></extra>"
                        ))

                    if not g_obs.empty:
                        fig.add_trace(go.Scatter(
                            x=g_obs["datetime"], y=g_obs["obs_temp"], mode="markers",
                            name=f"{d}m 実測", legendgroup=lg, showlegend=True,
                            marker=dict(size=6, color=emphasize_color(base_col), line=dict(color="black", width=0.1)),
                            opacity=0.80,
                            hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>実測水温: %{y:.2f} °C<extra></extra>"
                        ))

                else:
                    if not g_pred.empty:
                        x = g_pred["datetime"]; y_pred = g_pred["pred_temp"].astype(float)
                        fig.add_trace(go.Scatter(
                            x=x, y=y_pred, mode="lines",
                            name=f"{d}m 予測", legendgroup=lg, showlegend=True,
                            line=dict(color=base_col, width=2.0), opacity=1.0,
                            hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>水温: %{y:.2f} °C"
                        ))

                    if not g_obs.empty:
                        fig.add_trace(go.Scatter(
                            x=g_obs["datetime"], y=g_obs["obs_temp"], mode="markers",
                            name=f"{d}m 実測", legendgroup=lg, showlegend=True,
                            marker=dict(size=4, color=emphasize_color(base_col), line=dict(color="black", width=0.1)),
                            opacity=0.40,
                            hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>実測水温: %{y:.2f} °C<extra></extra>"
                        ))
        
            fig.update_layout(
                title={"text": f"{selected_file} 水温{title_suffix}", "y": 0.98, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
                margin=dict(l=10, r=10, t=50, b=10),
                height=550, template="plotly_white",
                legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1, font=dict(size=12))
            )

            x_range = [pd.Timestamp(start_day), pd.Timestamp(end_day) + pd.Timedelta(days=1)]
            fig.update_xaxes(type="date", range=x_range, title_text="日時（JST）", tickfont=dict(size=11))
            fig.update_yaxes(title_text="水温 (℃)", tickfont=dict(size=11))
            st.plotly_chart(fig, use_container_width=True)
   
    else:
        use_corr_bg = (corr_available and (not df_corr_period.empty) and ("corr_temp" in df_corr_period.columns))
        if use_corr_bg:
            bg_name = "補正"
            bg_df = df_corr_period[["datetime","depth_m","corr_temp"]].rename(columns={"corr_temp":"bg_temp"}).copy()
        else:
            bg_name = "予測"
            bg_df = df_period[["datetime","depth_m","pred_temp"]].rename(columns={"pred_temp":"bg_temp"}).copy()
    
        if bg_df.empty:
            st.warning("コンター表示できるデータがありません")
            st.stop()
    
        bg_df["depth_m"] = pd.to_numeric(bg_df["depth_m"], errors="coerce").round(0).astype("Int64")
        bg_df["bg_temp"] = pd.to_numeric(bg_df["bg_temp"], errors="coerce").astype(float).clip(lower=TEMP_MIN, upper=TEMP_MAX)
        bg_df["datetime"] = pd.to_datetime(bg_df["datetime"], errors="coerce")
        bg_df = bg_df.dropna(subset=["datetime","depth_m","bg_temp"]).copy()
        bg_df["time_bin"] = bg_df["datetime"].dt.floor(snap_freq)
        bg_df = bg_df.groupby(["depth_m","time_bin"], as_index=False)["bg_temp"].mean()
    
        depths_all = sorted(set(bg_df["depth_m"].dropna().astype(int).tolist()))
        if not depths_all:
            st.warning("深度情報がありません")
            st.stop()
   
        t0 = pd.Timestamp(start_day)
        t1 = pd.Timestamp(end_day)
        if snap_freq == "1h":
            time_grid = pd.date_range(t0, t1 + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq="1h")
        else:
            time_grid = pd.date_range(t0, t1, freq="1D")
   
        pv = (
            bg_df.pivot_table(index="depth_m", columns="time_bin", values="bg_temp", aggfunc="mean")
            .reindex(index=depths_all, columns=time_grid)
        )
        z = pv.values
        if z.size == 0 or (not np.isfinite(np.nanmax(z))):
            st.warning("コンター表示できる値がありません")
            st.stop()
    
        zmin = float(np.nanmin(z)); zmax = float(np.nanmax(z))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin == zmax:
            zmin, zmax = TEMP_MIN, TEMP_MAX
    
   
    if graph_style != "折れ線":
        site_id = os.path.splitext(selected_file)[0] if selected_file else ""
        df_thetao = load_cmem_thetao(site_id, fp="") if site_id else pd.DataFrame()
        thetao_ok = (isinstance(df_thetao, pd.DataFrame) and (not df_thetao.empty))
        diff_candidates = []
        if corr_available and (not df_corr_period.empty) and ("corr_temp" in df_corr_period.columns):
            diff_candidates.append("実測 − 補正")
        diff_candidates.append("実測 − 予測")
        if thetao_ok:
            diff_candidates.append("実測 − CMEM(thetao)")
        default_diff = ("実測 − 補正" if "実測 − 補正" in diff_candidates else "実測 − 予測")
        diff_mode = st.session_state.get("graph_diff_mode", default_diff)
        if diff_mode not in diff_candidates:
            diff_mode = default_diff
            st.session_state["graph_diff_mode"] = diff_mode

        tab_wt, tab_cum, tab_thr = st.tabs(["水温", "積算水温", "22℃基準"])
        def _render_wt_contour(_contour_value: str):
            contour_value = _contour_value
            if graph_style == "折れ線":
                diff_freq = "1h"
            else:
                diff_freq = (
                    "1h"
                    if ("graph_contour_agg" in st.session_state
                        and st.session_state.get("graph_contour_agg") == "1時間")
                    else "1D"
                )
        
            from plotly.subplots import make_subplots
            start_ts = pd.Timestamp(start_day)
            end_ts   = pd.Timestamp(end_day) + pd.Timedelta(days=1)
        
            try:
                full_times = pd.date_range(start_ts, end_ts, freq=diff_freq, inclusive='left')
            except TypeError:
                step = pd.Timedelta(hours=1) if diff_freq == "1h" else pd.Timedelta(days=1)
                full_times = pd.date_range(start_ts, end_ts - step, freq=diff_freq)
        
            use_corr_bg = (corr_available and not df_corr_period.empty and "corr_temp" in df_corr_period.columns)
            if use_corr_bg:
                bg_name = "補正"
                bg = df_corr_period.rename(columns={"corr_temp":"bg_temp"})[["datetime","depth_m","bg_temp"]].copy()
            else:
                bg_name = "予測"
                bg = df_period.rename(columns={"pred_temp":"bg_temp"})[["datetime","depth_m","bg_temp"]].copy()
        
            bg["time_bin"] = bg["datetime"].dt.floor(diff_freq)
            depths_bg = sorted(bg["depth_m"].dropna().astype(int).unique())
            pv_bg = (
                bg.pivot_table(index="depth_m", columns="time_bin", values="bg_temp", aggfunc="mean")
                  .reindex(index=depths_bg, columns=full_times)
            )
        
            # --- 下段：差分（選択） ---
            site_id = os.path.splitext(selected_file)[0] if selected_file else ""
            df_thetao = load_cmem_thetao(site_id, fp="") if site_id else pd.DataFrame()
            thetao_ok = (isinstance(df_thetao, pd.DataFrame) and (not df_thetao.empty))
            diff_candidates = []
            if corr_available and (not df_corr_period.empty) and ("corr_temp" in df_corr_period.columns):
                diff_candidates.append("実測 − 補正")
            diff_candidates.append("実測 − 予測")
            if thetao_ok:
                diff_candidates.append("実測 − CMEM(thetao)")
            default_diff = ("実測 − 補正" if "実測 − 補正" in diff_candidates else "実測 − 予測")
            diff_mode = st.session_state.get("graph_diff_mode", default_diff)
            if diff_mode not in diff_candidates:
                diff_mode = default_diff
                st.session_state["graph_diff_mode"] = diff_mode
            
            def _bin_series(df_src: pd.DataFrame, value_col: str, agg: str, out_col: str) -> pd.DataFrame:
                if not (isinstance(df_src, pd.DataFrame) and (not df_src.empty)):
                    return pd.DataFrame(columns=["depth_m", "time_bin", out_col])
                tmp = df_src.copy()
                if "datetime" not in tmp.columns:
                    return pd.DataFrame(columns=["depth_m", "time_bin", out_col])
                tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce")
                tmp["depth_m"] = pd.to_numeric(tmp.get("depth_m"), errors="coerce").round(0).astype("Int64")
                tmp[value_col] = pd.to_numeric(tmp.get(value_col), errors="coerce")
                tmp = tmp.dropna(subset=["datetime", "depth_m", value_col]).copy()
                tmp = tmp[(tmp["datetime"] >= start_ts) & (tmp["datetime"] < end_ts)].copy()
                if tmp.empty:
                    return pd.DataFrame(columns=["depth_m", "time_bin", out_col])
                tmp["time_bin"] = tmp["datetime"].dt.floor(diff_freq)
                if agg == "median":
                    out = tmp.groupby(["depth_m", "time_bin"], as_index=False)[value_col].median()
                else:
                    out = tmp.groupby(["depth_m", "time_bin"], as_index=False)[value_col].mean()
                out = out.rename(columns={value_col: out_col})
                return out
            
            obs_bin = _bin_series(df_obs, "obs_temp", ("median" if diff_freq == "1h" else "mean"), "obs")
            pred_bin = _bin_series(df_period, "pred_temp", "mean", "pred")
            corr_bin = _bin_series(df_corr_period, "corr_temp", "mean", "corr")
            thetao_bin = _bin_series(df_thetao, "thetao", "mean", "thetao") if thetao_ok else pd.DataFrame(columns=["depth_m","time_bin","thetao"])
            
            if diff_mode == "実測 − 補正":
                A, B = obs_bin, corr_bin; a_col, b_col = "obs", "corr"; a_lbl, b_lbl = "実測", "補正"
            elif diff_mode == "実測 − 予測":
                A, B = obs_bin, pred_bin; a_col, b_col = "obs", "pred"; a_lbl, b_lbl = "実測", "予測"
            elif diff_mode == "実測 − CMEM(thetao)":
                A, B = obs_bin, thetao_bin; a_col, b_col = "obs", "thetao"; a_lbl, b_lbl = "実測", "CMEM(thetao)"
            else:
                A, B = obs_bin, pred_bin; a_col, b_col = "obs", "pred"; a_lbl, b_lbl = "実測", "予測"
            diff_title = f"{a_lbl} − {b_lbl}"
            diff_title = f"{a_lbl} − {b_lbl}"
            
            _depths_for_grid = depths_bg
            _times_for_grid = list(full_times)
            grid = pd.DataFrame({
                "depth_m": np.repeat(_depths_for_grid, len(_times_for_grid)),
                "time_bin": np.tile(_times_for_grid, len(_depths_for_grid)),
            })
            mrg = grid.merge(A[["depth_m","time_bin",a_col]], on=["depth_m","time_bin"], how="left")
            mrg = mrg.merge(B[["depth_m","time_bin",b_col]], on=["depth_m","time_bin"], how="left")
            mrg["delta"] = mrg[a_col] - mrg[b_col]
        
            mrg2 = mrg.dropna(subset=["depth_m","time_bin"]).copy()
            mrg2["depth_m"] = mrg2["depth_m"].astype(int)
            depths_d = sorted(mrg2["depth_m"].unique())
            pv_d = (
                mrg2.pivot_table(index="depth_m", columns="time_bin", values="delta", aggfunc="mean")
                    .reindex(index=depths_d, columns=full_times)
            )
        
            z_bg = pv_bg.values
            z_d  = pv_d.values
        
            absmax = float(np.nanmax(np.abs(z_d))) if np.isfinite(np.nanmax(np.abs(z_d))) else 1.0
            absmax = max(absmax, 0.3)
        
            zmin_bg = float(np.nanmin(z_bg)) if np.isfinite(np.nanmin(z_bg)) else TEMP_MIN
            zmax_bg = float(np.nanmax(z_bg)) if np.isfinite(np.nanmax(z_bg)) else TEMP_MAX
            if (not np.isfinite(zmin_bg)) or (not np.isfinite(zmax_bg)) or (zmin_bg == zmax_bg):
                zmin_bg, zmax_bg = TEMP_MIN, TEMP_MAX
        
            cb1 = dict(title="℃", x=1.02, y=0.78, len=0.42, thickness=12)
            cb2 = dict(title="Δ℃", x=1.02, y=0.22, len=0.42, thickness=12)
        
            contour_agg_label = (
                contour_agg if "contour_agg" in locals() else ("1時間" if ("diff_freq" in locals() and diff_freq == "1h") else "日平均")
            )

            z_plot = z_bg
            zmin_plot, zmax_plot = zmin_bg, zmax_bg
            bg_title_name = "水温コンター"
            cb1_title = "℃"
            bg_colorscale = "Turbo"
            hover_bg = "日時=%{x|%Y-%m-%d %H:%M}<br>水深=%{y}m<br>T=%{z:.2f}℃<extra></extra>"

            if 'contour_value' in locals() and contour_value == "積算水温":
                if len(full_times) >= 2:
                    dt_days = (full_times[1] - full_times[0]).total_seconds() / 86400.0
                else:
                    dt_days = 1.0

                z_fill = np.where(np.isfinite(z_bg), z_bg, 0.0)
                z_plot = np.cumsum(z_fill * dt_days, axis=1)

                zmin_plot = 0.0
                zmax_plot = float(np.nanmax(z_plot)) if np.isfinite(np.nanmax(z_plot)) else 1.0
                bg_title_name = "積算水温コンター"
                cb1_title = "℃・day"
                hover_bg = "日時=%{x|%Y-%m-%d %H:%M}<br>水深=%{y}m<br>積算=%{z:.2f}℃・day<extra></extra>"
            
            elif 'contour_value' in locals() and contour_value == "22℃基準":
                z_plot = np.maximum(z_bg - HIGH_TEMP_TH, 0.0)
                zmin_plot = 0.0
                try:
                    zmax_plot = float(np.nanquantile(z_plot, 0.98))
                except Exception:
                    zmax_plot = float(np.nanmax(z_plot)) if np.isfinite(np.nanmax(z_plot)) else 0.5
                if (not np.isfinite(zmax_plot)) or (zmax_plot <= 0):
                    zmax_plot = 0.5
                zmax_plot = max(zmax_plot, 0.5)
                bg_title_name = "22℃基準温度コンター"
                cb1_title = "超過℃"
                bg_colorscale = "Reds"
                hover_bg = "日時=%{x|%Y-%m-%d %H:%M}<br>水深=%{y}m<br>基準超過=%{z:.2f}℃<extra></extra>"

            try:
                cb1['title'] = cb1_title
            except Exception:
                pass

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.56, 0.44], vertical_spacing=0.14,
                subplot_titles=(f"{bg_title_name}（{bg_name}・{contour_agg_label}）", f"差分（{diff_title}・{contour_agg_label}）")
            )
            fig.layout.annotations[1].update(yshift=20)
        
            fig.add_trace(go.Heatmap(
                x=full_times, y=depths_bg, z=z_plot,
                colorscale=bg_colorscale, zmin=zmin_plot, zmax=zmax_plot,
                zsmooth="best",  
                colorbar=cb1,
                hovertemplate=hover_bg
            ), row=1, col=1)

            if contour_value == "水温":
                zmin_iso = float(np.floor(np.nanmin(z_bg))) if np.isfinite(np.nanmin(z_bg)) else zmin_bg
                zmax_iso = float(np.ceil(np.nanmax(z_bg))) if np.isfinite(np.nanmax(z_bg)) else zmax_bg
                
                fig.add_trace(go.Contour(
                    x=full_times, y=depths_bg, z=z_bg,
                    contours=dict(
                        start=zmin_iso,
                        end=zmax_iso,
                        size=1.0,          # 1℃固定
                        coloring="none"
                    ),
                    line=dict(
                        color="rgba(0,0,0,0.35)",
                        width=1
                    ),
                    showscale=False,
                    hoverinfo="skip",
                    name="等温線（1℃）"
                ), row=1, col=1)

            if contour_value == "積算水温":
                zmax_cum = float(np.nanmax(z_plot)) if np.isfinite(np.nanmax(z_plot)) else 0.0
                if zmax_cum > 0:
                    fig.add_trace(go.Contour(
                        x=full_times, y=depths_bg, z=z_plot,
                        contours=dict(start=0.0, end=zmax_cum, size=100.0, coloring="none"),
                        line=dict(color="rgba(0,0,0,0.35)", width=1),
                        showscale=False, hoverinfo="skip",
                        name="等積算線（100℃・day）"
                    ), row=1, col=1)
        
            fig.add_trace(go.Heatmap(
                x=full_times, y=depths_d, z=z_d,
                colorscale="RdBu_r", zmin=-absmax, zmax=absmax,
                zsmooth="best",  
                colorbar=cb2,
                hovertemplate="日時=%{x|%Y-%m-%d %H:%M}<br>水深=%{y}m<br>Δ=%{z:.2f}℃<extra></extra>"
            ), row=2, col=1)
        
            fig.update_xaxes(type="date", range=[start_ts, end_ts], showticklabels=True, title_text=None, tickfont=dict(size=10), row=1, col=1)
            fig.update_xaxes(type="date", range=[start_ts, end_ts], title_text="日時（JST）", tickfont=dict(size=10), row=2, col=1)
        
            fig.update_yaxes(title_text="水深 (m)", autorange="reversed", row=1, col=1)
            fig.update_yaxes(title_text="水深 (m)", autorange="reversed", row=2, col=1)

            if getattr(fig.layout, "annotations", None):
                for a in fig.layout.annotations:
                    a.update(x=0.01, xanchor="left", font=dict(size=13), yshift=(-8 if "差分" in (a.text or "") else 0))
        
            fig.update_layout(
                height=760, template="plotly_white",
                margin=dict(l=55, r=95, t=55, b=40)
            )
        
            st.plotly_chart(fig, use_container_width=True)


        with tab_wt:
            _render_wt_contour("水温")
        with tab_cum:
            _render_wt_contour("積算水温")
        with tab_thr:
            _render_wt_contour("22℃基準")

        if isinstance(diff_candidates, list) and (len(diff_candidates) > 0):
            try:
                _cur = st.session_state.get("graph_diff_mode", diff_mode)
                _idx = diff_candidates.index(_cur) if _cur in diff_candidates else 0
            except Exception:
                _idx = 0
            st.selectbox("", diff_candidates, index=_idx, key="graph_diff_mode", label_visibility="collapsed")
