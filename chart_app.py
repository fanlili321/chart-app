#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指数走势图生成器 · 多子表版
运行方式：python3 -m streamlit run chart_app.py
"""

import io
import json
import zipfile

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.rcParams["font.sans-serif"] = ["Source Han Sans CN", "Heiti TC", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

AUTO_COLORS = ["#F17640", "#7CA7E0", "#929292", "#6674A6", "#FF2700", "#C24785", "#C8A028"]
COLOR_NAMES = {
    "#F17640": "① 橙色  #F17640",
    "#7CA7E0": "② 浅蓝  #7CA7E0",
    "#929292": "③ 灰色  #929292",
    "#6674A6": "④ 中蓝  #6674A6",
    "#FF2700": "⑤ 红色  #FF2700",
    "#C24785": "⑥ 玫红  #C24785",
    "#C8A028": "⑦ 金黄  #C8A028",
}

st.set_page_config(page_title="银行螺丝钉 ppt图表生成器", layout="wide", page_icon="📈")
st.markdown("""
<style>
/* ── 整体容器 ─────────────────────────────────────────────── */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px;
}
/* ── 文件上传框 ───────────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    border: 2px dashed #C8A028 !important;
    border-radius: 12px !important;
    background: #fffdf5 !important;
    padding: 6px 12px !important;
}
/* ── 主按钮 ───────────────────────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg,#B8920A,#D4A820) !important;
    border: none !important; color: #fff !important;
    font-weight: 700 !important; font-size: 15px !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 10px rgba(200,160,40,.35) !important;
    transition: all .2s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(90deg,#C8A028,#E8C060) !important;
    box-shadow: 0 5px 16px rgba(200,160,40,.5) !important;
    transform: translateY(-1px) !important;
}
/* ── 普通按钮 ─────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all .15s !important;
}
/* ── Expander ─────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    border: 1px solid #e2d9c0 !important;
    border-radius: 12px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.05) !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] > details > summary {
    background: #fdf9ee !important;
    padding: 13px 18px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
/* ── 输入框 ───────────────────────────────────────────────── */
.stTextInput input {
    border-radius: 7px !important;
    border-color: #d4c98a !important;
}
.stTextInput input:focus {
    border-color: #C8A028 !important;
    box-shadow: 0 0 0 2px rgba(200,160,40,.18) !important;
}
/* ── selectbox ────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 7px !important;
    border-color: #d4c98a !important;
}
/* ── 分割线 ───────────────────────────────────────────────── */
hr { border-color: #ede6d0 !important; margin: 22px 0 !important; }
/* ── success / info ───────────────────────────────────────── */
div[data-testid="stAlert"] { border-radius: 10px !important; }
/* ── caption ─────────────────────────────────────────────── */
.stCaption { color: #888 !important; }
</style>
""", unsafe_allow_html=True)

# ── 数据复核 ────────────────────────────────────────────────────────

def _detect_spikes(s: pd.Series) -> list[int]:
    """
    MAD 检测突变点：计算每期变化量，只有变化量偏离中位水平超过 5 倍 MAD
    且绝对变化超过数值量级 5% 时才标记，对趋势型数据不敏感。
    """
    if len(s) < 6:
        return []
    changes = s.diff().dropna()
    med_chg = changes.median()
    mad     = (changes - med_chg).abs().median()
    if mad < 1e-10:
        return []
    scale = s.abs().median()
    spikes = []
    for i, (_, chg) in enumerate(changes.items()):
        if abs(chg - med_chg) > 5 * mad and abs(chg) > scale * 0.05:
            spikes.append(i + 1)   # i+1：diff 从第1期开始
    return spikes


def _context_table(s: pd.Series, spike_idx: int, window: int = 3):
    """
    返回以 spike_idx 为中心、前后各 window 行的数据表，
    以及异常行在表中的行号（用于高亮）。
    """
    lo = max(0, spike_idx - window)
    hi = min(len(s), spike_idx + window + 1)
    sub = s.iloc[lo:hi].copy()
    df = pd.DataFrame({
        "日期": [
            idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            for idx in sub.index
        ],
        "数值": sub.round(4).values,
    })
    highlight_row = spike_idx - lo   # 异常行在子表中的位置
    return df, highlight_row


def render_data_check(sheets: dict):
    """数据复核：自动检测突变 + 跨表不一致，展示原始数据定位。"""
    try:
        _render_data_check_inner(sheets)
    except Exception as e:
        st.warning(f"数据复核出现错误（{e}），已跳过。")


def _render_data_check_inner(sheets: dict):
    # ── 准备 ──────────────────────────────────────────────────────────
    sheet_parsed = {}
    col_to_sheets = {}
    for sn, df in sheets.items():
        dc, vc = detect_date_and_value_cols(df)
        sheet_parsed[sn] = (dc, vc or [])
        for col in (vc or []):
            col_to_sheets.setdefault(col, []).append(sn)
    shared_cols = {col: sns for col, sns in col_to_sheets.items() if len(sns) > 1}

    # ── 检测突变 ──────────────────────────────────────────────────────
    # spike_issues: (sheet名, 列名, 数值Series, [异常行索引列表])
    spike_issues = []
    for sn, df in sheets.items():
        dc, vc = sheet_parsed[sn]
        if not dc or not vc:
            continue
        df_w = df.copy()
        df_w["__d__"] = pd.to_datetime(df_w[dc], errors="coerce")
        df_w = df_w.dropna(subset=["__d__"]).set_index("__d__").sort_index()
        for col in vc:
            try:
                s = pd.to_numeric(df_w[col], errors="coerce").dropna()
                idxs = _detect_spikes(s)
                if idxs:
                    spike_issues.append((sn, col, s, idxs))
            except Exception:
                continue

    # ── 检测跨表不一致 ────────────────────────────────────────────────
    cross_issues = []
    for col, sns in shared_cols.items():
        merged = None
        for sn in sns:
            df = sheets[sn]
            dc, vc = sheet_parsed[sn]
            if not dc or col not in vc:
                continue
            sub = df[[dc, col]].copy()
            sub[dc] = pd.to_datetime(sub[dc], errors="coerce")
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = (sub.dropna(subset=[dc, col])
                   .set_index(dc)[[col]]
                   .rename(columns={col: sn}))
            sub = sub[~sub.index.duplicated(keep="first")]
            merged = sub if merged is None else merged.join(sub, how="inner")
        if merged is None or len(merged.columns) < 2 or merged.empty:
            continue
        merged = merged.sort_index()

        def row_diff(row):
            v = pd.to_numeric(row, errors="coerce").dropna()
            if len(v) < 2:
                return 0.0
            ref = max(float(v.abs().max()), 1e-10)
            return float((v.max() - v.min()) / ref)

        diff_mask = merged.apply(row_diff, axis=1) > 0.01
        if diff_mask.any():
            cross_issues.append((col, merged, diff_mask))

    # ── 汇总 ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("⚠️ 数值突变", len(spike_issues))
    c2.metric("🔀 跨表不一致", len(cross_issues))
    c3.metric("已检查子表数", len(sheets))

    if not spike_issues and not cross_issues:
        st.success("✅ 未发现明显数据异常。")
        return

    # ── 展示突变：每处显示前后几行原始数据，异常行标红 ───────────────
    if spike_issues:
        with st.expander(f"⚠️ 数值突变（{len(spike_issues)} 处）", expanded=True):
            st.caption("下表显示异常点前后各3行的原始数据，🔴标红行即为疑似异常，请对照原始Excel核查。")
            for sn, col, s, idxs in spike_issues:
                st.markdown(f"**子表：{sn}　｜　列：{col}**")
                for spike_idx in idxs[:3]:   # 每列最多展示3处
                    ctx_df, hl_row = _context_table(s, spike_idx, window=3)
                    prev = round(float(s.iloc[spike_idx - 1]), 4) if spike_idx > 0 else None
                    curr = round(float(s.iloc[spike_idx]), 4)
                    dt   = ctx_df.iloc[hl_row]["日期"]
                    if prev is not None:
                        change_pct = (curr - prev) / abs(prev) * 100 if prev != 0 else 0
                        st.markdown(f"📍 **{dt}**：`{prev}` → `{curr}`（变化 {change_pct:+.1f}%）")
                    else:
                        st.markdown(f"📍 **{dt}**：`{curr}`")

                    def _hl_spike(row):
                        return ["background-color:#ffe0e0; font-weight:bold"
                                if row.name == hl_row else "" for _ in row]

                    st.dataframe(
                        ctx_df.style.apply(_hl_spike, axis=1),
                        use_container_width=True,
                        hide_index=True,
                        height=(len(ctx_df) + 1) * 35 + 10,
                    )
                st.markdown("---")

    # ── 展示跨表不一致：并排显示，差异行标红 ────────────────────────
    if cross_issues:
        with st.expander(f"🔀 跨表数据不一致（{len(cross_issues)} 列）", expanded=True):
            st.caption("同一日期、同名列在不同子表中数值不同，🔴标红行请核查数据来源是否一致。")
            for col, merged, diff_mask in cross_issues:
                st.markdown(f"**列：{col}**　（出现在子表：{' / '.join(merged.columns.tolist())}）")
                display = merged.copy()
                display.index = display.index.strftime("%Y-%m-%d")
                display.index.name = "日期"
                display = display.round(4)
                diff_dates_str = set(display.index[diff_mask.values])

                def _hl_cross(row):
                    bg = "background-color:#ffe0e0" if row.name in diff_dates_str else ""
                    return [bg] * len(row)

                st.dataframe(
                    display.style.apply(_hl_cross, axis=1),
                    use_container_width=True,
                    height=min(420, (len(display) + 1) * 35 + 10),
                )
                st.markdown("---")
                st.markdown("---")


# ── 工具函数 ────────────────────────────────────────────────────────

def find_header_row(raw: pd.DataFrame) -> int:
    best_row, best_count = 0, -1
    for i in range(min(20, len(raw))):
        row = raw.iloc[i].dropna()
        str_count = sum(isinstance(v, str) for v in row)
        if str_count > best_count:
            best_count = str_count
            best_row = i
    return best_row


def drop_sub_header(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    first = df.iloc[0].dropna()
    if len(first) == 0:
        return df
    str_ratio = sum(isinstance(v, str) for v in first) / len(first)
    if str_ratio >= 0.5:
        return df.iloc[1:].reset_index(drop=True)
    return df


def load_sheets(f) -> dict[str, pd.DataFrame]:
    name = f.name.lower()
    if name.endswith(".csv"):
        raw = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
            try:
                f.seek(0)
                raw = pd.read_csv(f, encoding=enc, header=None)
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            st.error("CSV 编码无法识别，请另存为 UTF-8 格式。")
            st.stop()
        header_row = find_header_row(raw)
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
            try:
                f.seek(0)
                df = pd.read_csv(f, encoding=enc, skiprows=header_row, header=0)
                break
            except Exception:
                continue
        return {"Sheet1": drop_sub_header(df)}
    elif name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(f)
        result = {}
        for sh in xl.sheet_names:
            raw = xl.parse(sh, header=None)
            header_row = find_header_row(raw)
            df = xl.parse(sh, skiprows=header_row, header=0)
            result[sh] = drop_sub_header(df)
        return result
    else:
        st.error("仅支持 .csv / .xlsx / .xls 格式。")
        st.stop()


def col_letter(n: int) -> str:
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def col_label(idx: int, col_name) -> str:
    return f"{col_letter(idx)}列: {col_name}"


def build_col_options(df: pd.DataFrame) -> dict[str, str]:
    return {col_label(i, c): c for i, c in enumerate(df.columns)}


def detect_date_and_value_cols(df: pd.DataFrame):
    date_col = None
    value_cols = []
    for col in df.columns:
        sample = df[col].dropna().head(10)
        if date_col is None:
            try:
                pd.to_datetime(sample, errors="raise")
                date_col = col
                continue
            except Exception:
                pass
        try:
            pd.to_numeric(sample, errors="raise")
            value_cols.append(col)
        except Exception:
            pass
    return date_col, value_cols


def adjust_labels(values: list, min_gap: float) -> list:
    if len(values) <= 1:
        return values[:]
    order = sorted(range(len(values)), key=lambda i: values[i])
    adj = [values[i] for i in order]
    for i in range(1, len(adj)):
        if adj[i] - adj[i - 1] < min_gap:
            adj[i] = adj[i - 1] + min_gap
    for i in range(len(adj) - 2, -1, -1):
        if adj[i + 1] - adj[i] < min_gap:
            adj[i] = adj[i + 1] - min_gap
    result = [0.0] * len(values)
    for rank, orig in enumerate(order):
        result[orig] = adj[rank]
    return result


def _compute_yticks(series_data_list: list):
    """从多条 Series 计算自适应 Y 轴刻度，返回 (tick_locs, y_lo, y_hi)。

    策略：先对数据上下各加 10% 的缓冲空间，再做取整计算刻度，
    确保走势线不贴着刻度上下边界，留有明显的视觉呼吸空间。
    """
    all_vals = np.concatenate([s.values for s in series_data_list])
    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    y_range = (y_max - y_min) or 1.0

    # 上下各扩展 10%，让走势线不贴着刻度边界
    pad = y_range * 0.10
    y_min_ext = y_min - pad
    y_max_ext = y_max + pad
    y_range_ext = y_max_ext - y_min_ext

    _unit  = float(10 ** np.floor(np.log10(y_range_ext / 10)))
    _y_lo  = float(np.floor(np.round(y_min_ext / _unit, 8)) * _unit)
    _y_hi  = float(np.ceil (np.round(y_max_ext / _unit, 8)) * _unit)
    _span  = (_y_hi - _y_lo) or 1.0

    def _nice_score(step):
        if step <= 0: return float("inf")
        mag  = 10 ** np.floor(np.log10(step))
        n    = step / mag
        nice = mag * (1 if n < 1.5 else 2 if n < 3.5 else 5 if n < 7.5 else 10)
        return abs(step / nice - 1)

    _best_n, _best_score = 8, float("inf")
    for _n in range(4, 9):
        score = _nice_score(_span / (_n - 1))
        if score < _best_score - 1e-9:
            _best_score = score
            _best_n = _n

    _n = _best_n
    _tick_locs = [round(_y_lo + i * _span / (_n - 1), 6) for i in range(_n)]
    _tick_locs[0]  = _y_lo
    _tick_locs[-1] = _y_hi
    # ylim 在刻度外再留 2% 的微小间隙，防止最高/最低刻度标签被裁剪
    y_lo = _y_lo - _span * 0.02
    y_hi = _y_hi + _span * 0.02
    return _tick_locs, y_lo, y_hi


FONT_SIZE    = 30
BORDER_COLOR = "#FFBF00"
PAD_H        = 0.022
PAD_V        = 0.022
SOURCE_GAP   = 0.010
LEG_FONT_MIN = 28

# 图表类型选项
SERIES_TYPES = {"折线图": "line", "柱状图": "bar", "面积图": "area"}
SERIES_TYPES_INV = {v: k for k, v in SERIES_TYPES.items()}

# Y轴归属选项
AXIS_OPTIONS = {"左轴": "left", "右轴": "right"}
AXIS_OPTIONS_INV = {v: k for k, v in AXIS_OPTIONS.items()}

# Y轴模式选项（左轴 & 右轴各自独立）
# percent   : 以首点为基准计算累计涨跌幅，显示 %
# points    : 显示原始数值，不加 % 符号（适合指数点数、价格等）
# points_pct: 显示原始数值，但加 % 符号（适合数据本身已是百分比，如国债收益率 4.5→ 4.5%）
YMODE_OPTIONS = {
    "涨跌幅（%）":      "percent",
    "指数点数":         "points",
    "原始值（带%标签）": "points_pct",
}
YMODE_OPTIONS_INV = {v: k for k, v in YMODE_OPTIONS.items()}


def _draw_one_series(target_ax, item: dict):
    """在指定坐标轴上绘制单条数据（折线 / 柱状 / 面积）。"""
    s     = item["data"]
    color = item["color"]
    stype = item.get("type", "line")

    if stype == "bar":
        # 根据数据间隔自动计算柱宽
        if len(s) >= 2:
            diffs = pd.Series(s.index).diff().dropna()
            width_days = max(1, int(diffs.median().days * 0.72))
        else:
            width_days = 30
        target_ax.bar(s.index, s.values, width=width_days,
                      color=color, alpha=0.45, zorder=2, linewidth=0)
    elif stype == "area":
        target_ax.fill_between(s.index, 0, s.values,
                               color=color, alpha=0.25, zorder=2)
        target_ax.plot(s.index, s.values, color=color, linewidth=2, zorder=3)
    else:  # line
        target_ax.plot(s.index, s.values, color=color, linewidth=4, zorder=3)


def build_figure(
    series_list: list,
    data_source: str = "数据来源：Wind",
    # y_mode / right_y_mode 已废弃，保留仅做兼容；实际从每条线的 mode 字段读取
    y_mode: str = "percent",
    right_y_mode: str = "points",
) -> plt.Figure:
    """
    series_list 每项字段：
      name   str         图例名称
      color  str         颜色 hex
      data   pd.Series   已经过日期筛选、索引为 DatetimeIndex
      type   str         'line' | 'bar' | 'area'       （默认 'line'）
      axis   str         'left' | 'right'              （默认 'left'）
      mode   str         'percent' | 'points' | 'points_pct'
                         percent    = 累计涨跌幅（以首点为基准换算，显示 %）
                         points     = 原始数值（不加 % 符号）
                         points_pct = 原始数值 + % 符号（数据本身已是百分比）
    """
    from matplotlib.transforms import Bbox

    has_right = any(s.get("axis", "left") == "right" for s in series_list)

    fig, ax = plt.subplots(figsize=(20, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax2 = ax.twinx() if has_right else None
    if ax2:
        ax2.set_facecolor("none")

    # ── 数据变换：每条线按自己的 mode 独立处理 ────────────────────────
    left_items, right_items = [], []
    for item in series_list:
        s    = item["data"].copy()
        axis = item.get("axis", "left")
        # 优先读取 item 自身的 mode；没有时按旧全局参数兼容
        mode = item.get("mode") or (y_mode if axis == "left" else right_y_mode)
        if mode == "percent":
            first = s.iloc[0]
            s = s * 100.0 if abs(first) < 1e-6 else (s / first - 1.0) * 100.0
        # points / points_pct：原始值不做换算
        processed = {**item, "data": s, "mode": mode}
        (left_items if axis == "left" else right_items).append(processed)

    # ── 绘图（柱状先画，确保折线在上层）────────────────────────────────
    for items, target_ax in [(left_items, ax), (right_items, ax2 or ax)]:
        for item in sorted(items, key=lambda x: 0 if x.get("type") == "bar" else 1):
            _draw_one_series(target_ax, item)

    # ── X 轴范围 ──────────────────────────────────────────────────────
    all_items = left_items + right_items
    all_dates = all_items[0]["data"].index
    for it in all_items[1:]:
        all_dates = all_dates.union(it["data"].index)
    date_range_days = (all_dates[-1] - all_dates[0]).days
    pad_days = max(int(date_range_days * 0.015), 3)
    ax.set_xlim(
        all_dates[0]  - pd.Timedelta(days=pad_days),
        all_dates[-1] + pd.Timedelta(days=pad_days),
    )

    # ── 左 Y 轴刻度 ───────────────────────────────────────────────────
    def _make_fmt(mode):
        """根据 y_mode 返回刻度格式化函数。"""
        def _fmt(x, _):
            close = abs(x - round(x)) < 0.05
            if mode in ("percent", "points_pct"):
                return f"{int(round(x))}%" if close else f"{x:.1f}%"
            return f"{int(round(x)):,}" if close else f"{x:,.1f}"
        return _fmt

    # 左轴格式：取左轴第一条线的 mode（同轴各线 mode 应一致）
    left_mode = left_items[0]["mode"] if left_items else "percent"
    if left_items:
        tick_locs, y_lo, y_hi = _compute_yticks([it["data"] for it in left_items])
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(mticker.FixedLocator(tick_locs))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_make_fmt(left_mode)))

    ax.tick_params(axis="y", labelsize=FONT_SIZE, length=0, pad=10,
                   labelleft=True, left=False)

    # ── 右 Y 轴刻度（完全独立于左轴，格式由右轴线条自身 mode 决定）────
    right_mode = right_items[0]["mode"] if right_items else "points"
    if ax2 and right_items:
        tick_locs_r, y_lo_r, y_hi_r = _compute_yticks([it["data"] for it in right_items])
        ax2.set_ylim(y_lo_r, y_hi_r)
        ax2.yaxis.set_major_locator(mticker.FixedLocator(tick_locs_r))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_make_fmt(right_mode)))
        ax2.tick_params(axis="y", labelsize=FONT_SIZE, length=0, pad=10,
                        labelright=True, labelleft=False, right=False, left=False)
        for spine in ax2.spines.values():
            spine.set_visible(False)

    # ── 网格（仅左轴） ────────────────────────────────────────────────
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.8, zorder=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # ── X 轴：仅首尾日期 ──────────────────────────────────────────────
    ax.set_xticks([all_dates[0], all_dates[-1]])
    ax.set_xticklabels(
        [all_dates[0].strftime("%Y-%m-%d"), all_dates[-1].strftime("%Y-%m-%d")],
        fontsize=FONT_SIZE, fontweight="bold", fontfamily="Source Han Sans CN",
    )
    ax.tick_params(axis="x", length=0, pad=10)

    # ── 边框线条：全隐藏 ──────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── 图例句柄（合并左右轴）────────────────────────────────────────
    legend_handles = []
    for item in series_list:
        stype = item.get("type", "line")
        if stype == "bar":
            h = mpatches.Patch(facecolor=item["color"], alpha=0.6, label=item["name"])
        elif stype == "area":
            h = plt.Line2D([0], [0], color=item["color"], linewidth=3,
                           label=item["name"])
        else:
            h = plt.Line2D([0], [0], color=item["color"], linewidth=4,
                           linestyle="-", label=item["name"])
        legend_handles.append(h)

    n = len(legend_handles)
    # 有右轴时右边预留空间，避免右轴刻度标签被裁剪
    right_margin = 0.90 if (ax2 and right_items) else 1.0
    fig.tight_layout(rect=[0.0, 0.10, right_margin, 1.0])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = ax.get_window_extent(renderer).width

    leg = None
    for try_ncols in range(n, 0, -1):
        leg = ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0, 1.10),
            frameon=False,
            handlelength=1.8,
            handleheight=0.8,
            ncol=try_ncols,
            columnspacing=1.2,
            labelspacing=0.4,
            borderpad=0,
            prop={"family": "Source Han Sans CN", "size": max(FONT_SIZE, LEG_FONT_MIN)},
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if leg.get_window_extent(renderer).width <= ax_w_px or try_ncols == 1:
            break

    # ── 强制所有 Y 轴刻度标签可见 ────────────────────────────────────
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_visible(True)
        tick.label1.set_fontfamily("Source Han Sans CN")
        tick.label1.set_fontsize(FONT_SIZE)

    if ax2 and right_items:
        fig.canvas.draw()
        for tick in ax2.yaxis.get_major_ticks():
            tick.label2.set_visible(True)
            tick.label2.set_fontfamily("Source Han Sans CN")
            tick.label2.set_fontsize(FONT_SIZE)
            tick.label1.set_visible(False)   # 确保右轴不在左侧重复显示

    # ── X 轴日期对齐 ─────────────────────────────────────────────────
    xlabels = ax.get_xticklabels()
    if len(xlabels) >= 1:
        xlabels[0].set_ha("left")
    if len(xlabels) >= 2:
        xlabels[-1].set_ha("right")

    # ── 最终 draw ────────────────────────────────────────────────────
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ── 黄色外框：紧包所有内容 ────────────────────────────────────────
    bboxes = [ax.get_tightbbox(renderer), leg.get_window_extent(renderer)]
    if ax2 and right_items:
        bb2 = ax2.get_tightbbox(renderer)
        if bb2:
            bboxes.append(bb2)
    full_bbox = Bbox.union(bboxes)

    inv = fig.transFigure.inverted()
    bf  = full_bbox.transformed(inv)
    bx  = bf.x0 - PAD_H
    by  = bf.y0 - PAD_V
    bw  = bf.width  + 2 * PAD_H
    bh  = bf.height + 2 * PAD_V

    fig.add_artist(mpatches.FancyBboxPatch(
        (bx, by), bw, bh,
        transform=fig.transFigure,
        boxstyle="square,pad=0",
        linewidth=4, edgecolor=BORDER_COLOR, facecolor="none", zorder=10,
    ))

    fig.text(
        bx + bw, by - SOURCE_GAP,
        data_source,
        ha="right", va="top",
        color="#999999",
        fontproperties=matplotlib.font_manager.FontProperties(
            family="Source Han Sans CN", size=FONT_SIZE
        ),
    )

    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()


def generate_one(sheet_name: str, cfg: dict, df: pd.DataFrame):
    """根据配置生成单张图，返回 (img_bytes, error_msg)。"""
    df = df.copy()
    df["__date__"] = pd.to_datetime(df[cfg["date_col"]], errors="coerce")
    df = df.dropna(subset=["__date__"]).set_index("__date__").sort_index()
    try:
        if cfg["date_start"]:
            df = df.loc[pd.Timestamp(cfg["date_start"]):]
        if cfg["date_end"]:
            df = df.loc[:pd.Timestamp(cfg["date_end"])]
    except Exception:
        pass
    if df.empty:
        return None, "筛选后无数据，请检查日期区间。"

    series_list = []
    for s_cfg in cfg["series"]:
        col = s_cfg["col"]
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            series_list.append({
                "name":  s_cfg["name"],
                "color": s_cfg["color"],
                "type":  s_cfg.get("type", "line"),
                "axis":  s_cfg.get("axis", "left"),
                "mode":  s_cfg.get("mode", cfg.get("y_mode", "percent")),  # 每条线自己的格式
                "data":  s,
            })
    if not series_list:
        return None, "无有效数据，请检查数据列。"

    fig = build_figure(
        series_list,
        cfg.get("data_source", "数据来源：Wind"),
    )
    img_bytes = fig_to_bytes(fig)
    plt.close(fig)
    return img_bytes, None


# ══════════════════════════════════════════════════════════════════
#  页面
# ══════════════════════════════════════════════════════════════════

def section_title(num: str, text: str):
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:28px 0 14px;">
  <div style="background:#C8A028;color:#fff;width:32px;height:32px;border-radius:50%;
              display:flex;align-items:center;justify-content:center;
              font-weight:800;font-size:15px;flex-shrink:0;
              box-shadow:0 2px 8px rgba(200,160,40,.4);">{num}</div>
  <div style="font-size:17px;font-weight:700;color:#1C2B4A;">{text}</div>
</div>""", unsafe_allow_html=True)

# ── 顶部横幅 ────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1C2B4A 0%,#243660 100%);
            border-radius:14px;padding:28px 40px;margin-bottom:24px;
            border-bottom:4px solid #C8A028;
            box-shadow:0 4px 20px rgba(28,43,74,.18);">
  <div style="color:#C8A028;font-size:11px;letter-spacing:3px;
              text-transform:uppercase;margin-bottom:8px;font-weight:600;">
    📈 &nbsp;Financial Index Chart Generator
  </div>
  <div style="color:#fff;font-size:24px;font-weight:700;
              letter-spacing:.5px;margin-bottom:8px;">
    银行螺丝钉 &nbsp;ppt图表生成器
  </div>
  <div style="color:rgba(255,255,255,.5);font-size:13px;line-height:1.8;">
    上传 Excel 多子表数据 &nbsp;·&nbsp; 逐图配置参数 &nbsp;·&nbsp; 一键生成专业走势图
  </div>
</div>
""", unsafe_allow_html=True)

# ── 配置保存 / 加载 ─────────────────────────────────────────────────
with st.expander("💾 保存 / 加载配置", expanded=False):
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        st.markdown("**保存当前配置**")
        save_name = st.text_input("配置名称", value="我的配置", key="save_name",
                                  help="给这套配置起个名字，方便识别")
        if st.button("💾 保存配置", use_container_width=True):
            if "all_configs" in st.session_state and "selected_sheets" in st.session_state:
                saved = {
                    "name": save_name,
                    "selected_sheets": st.session_state["selected_sheets"],
                    "configs": st.session_state["all_configs"],
                }
                json_bytes = json.dumps(saved, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    label=f"⬇️ 下载 {save_name}.json",
                    data=json_bytes,
                    file_name=f"{save_name}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.warning("请先上传文件并完成配置，再保存。")

    with cfg_col2:
        st.markdown("**加载已有配置**")
        cfg_file = st.file_uploader("上传配置文件（.json）", type=["json"],
                                    key="cfg_upload", label_visibility="collapsed")
        if cfg_file is not None:
            try:
                loaded_cfg = json.loads(cfg_file.read().decode("utf-8"))
                st.session_state["loaded_cfg"] = loaded_cfg
                st.success(f"已加载配置：{loaded_cfg.get('name', '未命名')}")
            except Exception as e:
                st.error(f"配置文件解析失败：{e}")

st.markdown("---")

# ── 上传 ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "上传数据文件（Excel 多子表 / CSV）",
    type=["xlsx", "xls", "csv"],
    label_visibility="visible",
)

if not uploaded:
    st.info("上传 Excel 文件，每个子表对应一张图，上传后选择要生成的子表并分别配置参数，一键生成全部图表。")
    with st.expander("文件格式说明"):
        st.markdown("""
**Excel 多子表格式（推荐）**
- 每个 Sheet = 一张图，Sheet 名称 = 输出文件名
- 第一列为日期，其余列为指数数据（每列一条线）

| 日期 | 中证全指 | 沪深300 |
|------|---------|---------|
| 2020-01-02 | 4521.3 | 3912.0 |

**Y轴模式**
- **涨跌幅（%）**：以第一个交易日为基准，显示累计涨跌幅
- **指数点数**：显示原始数值

**图表类型（每条线可单独设置）**
- **折线图**：标准走势线
- **面积图**：填充色底 + 走势线，适合做背景层（如图一的国债收益率）
- **柱状图**：柱形展示，适合年度/季度数据（如图二的归母净利润）

**双Y轴**
- 每条线可独立选择归属左轴或右轴，左右轴各自独立刻度
        """)
    st.stop()

# ── 读取所有子表 ─────────────────────────────────────────────────────
sheets = load_sheets(uploaded)
sheet_names = list(sheets.keys())
st.success(f"已读取 **{len(sheet_names)}** 个子表：{', '.join(f'`{s}`' for s in sheet_names)}")

with st.expander("🔍 数据预览（确认列名是否正确识别）", expanded=False):
    for sn in sheet_names[:4]:
        st.caption(f"子表：{sn}")
        st.dataframe(sheets[sn].head(), use_container_width=True)

# ── 数据复核 ─────────────────────────────────────────────────────────
st.markdown("---")
section_title("1", "数据复核")
st.caption("自动检测：数值突增突降 · 跨表同列数据不一致")
render_data_check(sheets)

# ── 第二步：作图 ─────────────────────────────────────────────────────
st.markdown("---")
section_title("2", "作图")
st.markdown("**① 选择子表**")

_loaded = st.session_state.get("loaded_cfg", {})
_loaded_sheets = [s for s in _loaded.get("selected_sheets", sheet_names) if s in sheet_names]
_default_selected = _loaded_sheets if _loaded_sheets else sheet_names

selected_sheets = st.multiselect(
    "勾选需要生成图表的子表（未勾选的直接跳过）",
    options=sheet_names,
    default=_default_selected,
    label_visibility="collapsed",
)
st.session_state["selected_sheets"] = selected_sheets

if not selected_sheets:
    st.warning("请至少选择一个子表。")
    st.stop()

# ── 逐表配置 ─────────────────────────────────────────────────────────
st.markdown("**② 配置每张图表**")

all_configs = {}
_loaded_configs = _loaded.get("configs", {})

for idx, sheet_name in enumerate(selected_sheets):
    df = sheets[sheet_name]
    auto_date, auto_vals = detect_date_and_value_cols(df)

    saved = _loaded_configs.get(sheet_name, {})

    date_series = None
    if auto_date:
        date_series = pd.to_datetime(df[auto_date], errors="coerce").dropna()

    col_opts   = build_col_options(df)
    label_of   = {v: k for k, v in col_opts.items()}
    all_labels = list(col_opts.keys())

    with st.expander(f"📊 图{idx+1}：{sheet_name}", expanded=True):

        # ── 行1：日期列 / 数据列 / 左轴模式 ─────────────────────────
        row1 = st.columns([2, 4, 2])

        with row1[0]:
            saved_date_col   = saved.get("date_col")
            saved_date_label = label_of.get(saved_date_col) if saved_date_col else None
            auto_date_label  = saved_date_label or label_of.get(auto_date, all_labels[0])
            date_label = st.selectbox(
                "日期列", all_labels,
                index=all_labels.index(auto_date_label) if auto_date_label in all_labels else 0,
                key=f"date_{sheet_name}",
            )
            date_col = col_opts[date_label]

        with row1[1]:
            non_date_labels = [l for l in all_labels if l != date_label]
            saved_val_cols   = [s["col"] for s in saved.get("series", [])]
            saved_val_labels = [label_of[c] for c in saved_val_cols
                                if c in label_of and label_of[c] in non_date_labels]
            auto_val_labels  = [label_of[c] for c in auto_vals
                                if c in label_of and label_of[c] != date_label]
            default_labels   = saved_val_labels or auto_val_labels or non_date_labels[:1]
            value_labels = st.multiselect(
                "数据列（每列一条线）", non_date_labels,
                default=default_labels,
                key=f"vals_{sheet_name}",
            )
            value_cols = [col_opts[l] for l in value_labels]

        with row1[2]:
            # 占位，不再用全局左轴模式（已移到每条线各自配置）
            st.caption("数值格式在各线条设置里独立配置 ↓")

        # ── 行2：日期区间 ────────────────────────────────────────────
        st.caption("日期区间（留空则使用数据中的全部日期）")
        date_row = st.columns([2, 2, 4])

        if date_series is not None and len(date_series) > 0:
            auto_start = date_series.min().strftime("%Y-%m-%d")
            auto_end   = date_series.max().strftime("%Y-%m-%d")
        else:
            auto_start = auto_end = ""

        with date_row[0]:
            date_start_str = st.text_input(
                "起始日期", value=saved.get("date_start") or auto_start,
                placeholder="如 2020-01-01", key=f"ds_{sheet_name}",
            )
        with date_row[1]:
            date_end_str = st.text_input(
                "结束日期", value=saved.get("date_end") or auto_end,
                placeholder="如 2024-12-31", key=f"de_{sheet_name}",
            )

        # ── 行3：线条详细设置（名称 / 颜色 / 图表类型 / Y轴归属） ────
        if value_cols:
            st.caption("线条设置（图例名称 · 颜色 · 图表类型 · Y轴归属）")
            sub_cols   = st.columns(min(len(value_cols), 3))
            series_cfg = []
            has_any_right = False

            for vi, vcol in enumerate(value_cols):
                default_color = AUTO_COLORS[vi % len(AUTO_COLORS)]
                with sub_cols[vi % len(sub_cols)]:
                    saved_series = {s["col"]: s for s in saved.get("series", [])}
                    saved_s      = saved_series.get(vcol, {})
                    saved_name   = saved_s.get("name",  vcol)
                    saved_color  = saved_s.get("color", default_color)
                    saved_type   = saved_s.get("type",  "line")
                    saved_axis   = saved_s.get("axis",  "left")
                    saved_mode   = saved_s.get("mode",  "percent")

                    st.markdown(f"**线条 {vi+1}**　`{vcol}`")

                    name = st.text_input(
                        "图例名称", value=saved_name,
                        key=f"name_{sheet_name}_{vi}",
                    )

                    safe_color = saved_color if saved_color in AUTO_COLORS else default_color
                    color = st.selectbox(
                        "线条颜色", options=AUTO_COLORS,
                        format_func=lambda c: COLOR_NAMES.get(c, c),
                        index=AUTO_COLORS.index(safe_color),
                        key=f"color_{sheet_name}_{vi}",
                    )
                    swatches = "".join(
                        f'<span title="{COLOR_NAMES.get(c, c)}" style="'
                        f'display:inline-block;width:22px;height:22px;'
                        f'background:{c};border-radius:4px;margin-right:4px;'
                        f'border:{"3px solid #333" if c == color else "2px solid #ccc"};'
                        f'"></span>'
                        for c in AUTO_COLORS
                    )
                    st.markdown(
                        f'<div style="margin-top:-6px;margin-bottom:6px">{swatches}</div>',
                        unsafe_allow_html=True,
                    )

                    # 图表类型
                    type_options = list(SERIES_TYPES.keys())
                    saved_type_label = SERIES_TYPES_INV.get(saved_type, "折线图")
                    series_type = st.selectbox(
                        "图表类型",
                        options=type_options,
                        index=type_options.index(saved_type_label),
                        key=f"type_{sheet_name}_{vi}",
                    )
                    series_type_val = SERIES_TYPES[series_type]

                    # Y轴归属 + 数值格式（同行）
                    _ax_col, _fmt_col = st.columns(2)
                    with _ax_col:
                        axis_options = list(AXIS_OPTIONS.keys())
                        saved_axis_label = AXIS_OPTIONS_INV.get(saved_axis, "左轴")
                        series_axis = st.radio(
                            "Y轴归属", axis_options,
                            index=axis_options.index(saved_axis_label),
                            key=f"axis_{sheet_name}_{vi}",
                            horizontal=True,
                        )
                        series_axis_val = AXIS_OPTIONS[series_axis]
                        if series_axis_val == "right":
                            has_any_right = True

                    with _fmt_col:
                        _mode_keys = list(YMODE_OPTIONS.keys())
                        _mode_vals = list(YMODE_OPTIONS.values())
                        _mode_idx  = _mode_vals.index(saved_mode) if saved_mode in _mode_vals else 0
                        series_mode_label = st.selectbox(
                            "数值格式",
                            options=_mode_keys,
                            index=_mode_idx,
                            key=f"mode_{sheet_name}_{vi}",
                            help="涨跌幅%：以首点为基准换算累计收益\n指数点数：显示原始数值\n原始值（带%）：数据本身已是%，如国债收益率4.5→显示4.5%",
                        )
                        series_mode_val = YMODE_OPTIONS[series_mode_label]

                    series_cfg.append({
                        "col":   vcol,
                        "name":  name,
                        "color": color,
                        "type":  series_type_val,
                        "axis":  series_axis_val,
                        "mode":  series_mode_val,
                    })

            # 数值格式已在每条线各自配置，此处无需全局右轴模式选项

        else:
            series_cfg   = []
            right_y_mode = "points"
            st.warning("请选择至少一列数据。")

        # ── 数据来源 ─────────────────────────────────────────────────
        data_source = st.text_input(
            "数据来源",
            value=saved.get("data_source", "数据来源：Wind"),
            key=f"src_{sheet_name}",
            help="默认「数据来源：Wind」，可追加截止日期等信息",
        )

        cfg_now = {
            "date_col":    date_col,
            "series":      series_cfg,   # 每条线含 mode 字段
            "y_mode":      "percent",    # 兼容旧配置保留，build_figure 优先用 series[].mode
            "date_start":  date_start_str.strip(),
            "date_end":    date_end_str.strip(),
            "data_source": data_source.strip(),
        }
        all_configs[sheet_name] = cfg_now

        # ── 单张生成按钮 ──────────────────────────────────────────────
        st.markdown("---")
        if st.button(f"🖼️ 生成此图", key=f"gen1_{sheet_name}", use_container_width=True):
            if not cfg_now["series"]:
                st.warning("请先选择数据列。")
            else:
                with st.spinner("生成中…"):
                    img_bytes, err = generate_one(sheet_name, cfg_now, sheets[sheet_name])
                if err:
                    st.error(err)
                else:
                    st.session_state[f"preview_{sheet_name}"] = img_bytes

        if f"preview_{sheet_name}" in st.session_state:
            _b = st.session_state[f"preview_{sheet_name}"]
            st.image(_b, use_container_width=True)
            st.download_button(
                f"⬇️ 下载 {sheet_name}.png", data=_b,
                file_name=f"{sheet_name}.png", mime="image/png",
                key=f"dl1_{sheet_name}", use_container_width=True,
            )

st.session_state["all_configs"] = all_configs

# ── 批量生成 ─────────────────────────────────────────────────────────
st.markdown("**③ 生成图表**")
generate_btn = st.button(
    f"🚀 一键生成全部图表（共 {len(selected_sheets)} 张）",
    type="primary",
    use_container_width=True,
)

if generate_btn:
    results  = []
    progress = st.progress(0, text="生成中...")
    total    = len(selected_sheets)
    done     = 0

    for sheet_name, cfg in all_configs.items():
        if not cfg["series"]:
            st.warning(f"「{sheet_name}」未选择数据列，已跳过。")
            done += 1
            continue

        img_bytes, err = generate_one(sheet_name, cfg, sheets[sheet_name])
        if err:
            st.warning(f"「{sheet_name}」{err}")
        else:
            st.session_state[f"preview_{sheet_name}"] = img_bytes
            results.append({"name": f"{sheet_name}.png", "bytes": img_bytes})
        done += 1
        progress.progress(done / total, text=f"已生成 {done}/{total}：{sheet_name}")

    progress.empty()

    if not results:
        st.error("没有成功生成任何图表，请检查数据。")
    else:
        st.success(f"✅ 成功生成 {len(results)} 张图表")

        if len(results) > 1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for r in results:
                    zf.writestr(r["name"], r["bytes"])
            zip_buf.seek(0)
            st.download_button(
                "⬇️ 打包下载全部图表（ZIP）",
                data=zip_buf,
                file_name=f"{uploaded.name.rsplit('.', 1)[0]}_charts.zip",
                mime="application/zip",
                use_container_width=True,
            )

        st.markdown("---")

        cols_per_row = 2
        for i in range(0, len(results), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, r in enumerate(results[i:i+cols_per_row]):
                with row_cols[j]:
                    st.image(r["bytes"], caption=r["name"], use_container_width=True)
                    st.download_button(
                        f"⬇️ {r['name']}", data=r["bytes"],
                        file_name=r["name"], mime="image/png",
                        key=f"dl_{r['name']}_{i}_{j}",
                        use_container_width=True,
                    )
