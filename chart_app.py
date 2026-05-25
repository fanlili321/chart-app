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

st.set_page_config(page_title="银行螺丝钉 指数走势生成器", layout="wide", page_icon="📈")
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

# ── 工具函数 ────────────────────────────────────────────────────────

def find_header_row(raw: pd.DataFrame) -> int:
    """
    扫描前20行，找字符串数量最多的行作为标题行。
    这样能跳过只有少量文字的说明行，找到列名最完整的那行。
    """
    best_row, best_count = 0, -1
    for i in range(min(20, len(raw))):
        row = raw.iloc[i].dropna()
        str_count = sum(isinstance(v, str) for v in row)
        if str_count > best_count:
            best_count = str_count
            best_row = i
    return best_row


def drop_sub_header(df: pd.DataFrame) -> pd.DataFrame:
    """如果数据第一行大多数值仍是字符串（子标题行），就跳过它。"""
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
    """读取 Excel 所有子表，自动识别标题行，返回 {sheet_name: DataFrame}。"""
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
    """将列索引（0起）转为 Excel 列字母，如 0→A, 25→Z, 26→AA。"""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def col_label(idx: int, col_name) -> str:
    """生成 'A列: 日期' 格式的标签。"""
    return f"{col_letter(idx)}列: {col_name}"


def build_col_options(df: pd.DataFrame) -> dict[str, str]:
    """返回 {标签: 原始列名} 的有序字典，供 selectbox/multiselect 使用。"""
    return {col_label(i, c): c for i, c in enumerate(df.columns)}


def detect_date_and_value_cols(df: pd.DataFrame):
    """自动识别日期列和数值列，返回原始列名。"""
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


FONT_SIZE  = 30
BORDER_COLOR = "#FFBF00"
PAD_H      = 0.022   # 左右边框留白（对称）
PAD_V      = 0.022   # 上下边框留白（对称）
SOURCE_GAP = 0.010   # 数据来源与边框底部间距

def build_figure(series_list: list, y_mode: str, data_source: str = "数据来源：Wind") -> plt.Figure:
    from matplotlib.transforms import Bbox

    fig, ax = plt.subplots(figsize=(20, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── 画线 ──────────────────────────────────────────────────────────
    plot_series = []
    for item in series_list:
        s = item["data"].copy()
        if y_mode == "percent":
            first = s.iloc[0]
            s = s * 100.0 if abs(first) < 1e-6 else (s / first - 1.0) * 100.0
        ax.plot(s.index, s.values, color=item["color"], linewidth=4, zorder=3)
        plot_series.append(s)

    # ── X 轴左右延伸：让走势线两端与网格线对齐 ───────────────────────
    dates = plot_series[0].index
    date_range_days = (dates[-1] - dates[0]).days
    pad_days = max(int(date_range_days * 0.015), 3)   # 约1.5%区间宽度
    ax.set_xlim(
        dates[0]  - pd.Timedelta(days=pad_days),
        dates[-1] + pd.Timedelta(days=pad_days),
    )

    # ── Y 轴范围 & 刻度：先算刻度位置，再由刻度推导 ylim ────────────
    # 用 MaxNLocator(nbins=10) 直接对数据极值计算刻度，
    # 步长会更细（如 10% 而非 20%），顶刻度紧贴数据最大值。
    all_vals = np.concatenate([s.values for s in plot_series])
    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))

    # ── Y 轴刻度自适应算法 ────────────────────────────────────────────
    # 流程：
    # 1. y_lo/y_hi = 圆整边界（略小于数据最小值 / 略大于数据最大值）
    # 2. 在 4~8 个刻度里，选让步长最接近"好看数字"(1/2/5×10^n)的个数
    # 3. 均匀分配，步长允许小数
    y_range = (y_max - y_min) or 1.0
    _unit  = float(10 ** np.floor(np.log10(y_range / 10)))
    _y_lo  = float(np.floor(np.round(y_min / _unit, 8)) * _unit)
    _y_hi  = float(np.ceil (np.round(y_max / _unit, 8)) * _unit)
    _span  = (_y_hi - _y_lo) or 1.0

    def _nice_score(step):
        """步长离最近的好看数字（1/2/5×10^n）越近，得分越低（越好）。"""
        if step <= 0:
            return float("inf")
        mag  = 10 ** np.floor(np.log10(step))
        n    = step / mag
        nice = mag * (1 if n < 1.5 else 2 if n < 3.5 else 5 if n < 7.5 else 10)
        return abs(step / nice - 1)

    # 从 4~8 中选得分最低（步长最"好看"）的刻度数
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

    # ylim 在 y_lo/y_hi 外各留 4% span 的呼吸空间（标签不贴边）
    y_lo = _y_lo - _span * 0.04
    y_hi = _y_hi + _span * 0.04

    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(mticker.FixedLocator(_tick_locs))

    # ── Y 轴格式 ──────────────────────────────────────────────────────
    def _fmt(x, _):
        close_to_int = abs(x - round(x)) < 0.05
        if y_mode == "percent":
            return f"{int(round(x))}%" if close_to_int else f"{x:.1f}%"
        else:
            return f"{int(round(x)):,}" if close_to_int else f"{x:,.1f}"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
    ax.tick_params(axis="y", labelsize=FONT_SIZE, length=0, pad=10,
                   labelleft=True, left=False)

    # ── 网格 ──────────────────────────────────────────────────────────
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.8, zorder=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # ── X 轴：仅首尾日期 ───────────────────────────────────────────────
    ax.set_xticks([dates[0], dates[-1]])
    ax.set_xticklabels(
        [dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
        fontsize=FONT_SIZE, fontweight="bold", fontfamily="Source Han Sans CN",
    )
    ax.tick_params(axis="x", length=0, pad=10)

    # ── 边框线条：四边全隐藏，视觉边界由黄色外框承担 ─────────────────
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── 图例：先定型坐标轴，再自适应列数，保证不超出图表右边界 ────────
    legend_handles = [
        plt.Line2D([0], [0], color=it["color"], linewidth=4, linestyle="-", label=it["name"])
        for it in series_list
    ]
    n = len(legend_handles)
    LEG_FONT_MIN = 28          # 图例字体下限

    # 先 tight_layout + draw，确定坐标轴的实际像素宽度
    fig.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = ax.get_window_extent(renderer).width

    # 从"全部放一行"开始，逐步减少列数，直到图例宽度 ≤ 坐标轴宽度
    # 如果减到 1 列还超出（标签本身超长），维持 1 列；字号保持 FONT_SIZE
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
            break   # 已合适，或已压到 1 列，停止

    # ── 强制所有 Y 轴刻度标签可见，并应用字体 ────────────────────────
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_visible(True)
        tick.label1.set_fontfamily("Source Han Sans CN")
        tick.label1.set_fontsize(FONT_SIZE)

    # ── X 轴日期对齐 ─────────────────────────────────────────────────
    xlabels = ax.get_xticklabels()
    if len(xlabels) >= 1:
        xlabels[0].set_ha("left")
    if len(xlabels) >= 2:
        xlabels[-1].set_ha("right")

    # ── 最终 draw：让字体和对齐生效，再取精确 bbox ───────────────────
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ── 获取所有内容的实际边界（坐标轴 + 刻度标签 + 图例） ────────────
    ax_bbox  = ax.get_tightbbox(renderer)
    leg_bbox = leg.get_window_extent(renderer)
    full_bbox = Bbox.union([ax_bbox, leg_bbox])

    # 转换为 figure 坐标（0~1）
    inv = fig.transFigure.inverted()
    bf = full_bbox.transformed(inv)

    bx = bf.x0 - PAD_H
    by = bf.y0 - PAD_V
    bw = bf.width  + 2 * PAD_H
    bh = bf.height + 2 * PAD_V

    # ── 黄色边框，紧包所有内容 ────────────────────────────────────────
    fig.add_artist(mpatches.FancyBboxPatch(
        (bx, by), bw, bh,
        transform=fig.transFigure,
        boxstyle="square,pad=0",
        linewidth=4, edgecolor=BORDER_COLOR, facecolor="none", zorder=10,
    ))

    # ── 数据来源：边框右下角正下方 ────────────────────────────────────
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
            series_list.append({"name": s_cfg["name"], "color": s_cfg["color"], "data": s})
    if not series_list:
        return None, "无有效数据，请检查数据列。"
    fig = build_figure(series_list, cfg["y_mode"], cfg.get("data_source", "数据来源：Wind"))
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
    银行螺丝钉 &nbsp;指数走势生成器
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
        """)
    st.stop()

# ── 读取所有子表（自动识别标题行） ─────────────────────────────────
sheets = load_sheets(uploaded)
sheet_names = list(sheets.keys())
st.success(f"已读取 **{len(sheet_names)}** 个子表：{', '.join(f'`{s}`' for s in sheet_names)}")

# 数据预览
with st.expander("🔍 数据预览（确认列名是否正确识别）", expanded=False):
    for sn in sheet_names[:4]:
        st.caption(f"子表：{sn}")
        st.dataframe(sheets[sn].head(), use_container_width=True)

# ── 第一步：选择要生成图表的子表 ────────────────────────────────────
st.markdown("---")
section_title("1", "选择要生成图表的子表")

# 从已加载配置中恢复子表选择
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

# ── 第二步：逐表配置 ─────────────────────────────────────────────────
st.markdown("---")
section_title("2", "配置每张图表")

all_configs = {}
_loaded_configs = _loaded.get("configs", {})

for idx, sheet_name in enumerate(selected_sheets):
    df = sheets[sheet_name]
    auto_date, auto_vals = detect_date_and_value_cols(df)

    # 从已加载配置中取该子表的历史配置（如有）
    saved = _loaded_configs.get(sheet_name, {})

    # 预先解析日期列，用于推断默认日期范围
    date_series = None
    if auto_date:
        date_series = pd.to_datetime(df[auto_date], errors="coerce").dropna()

    col_opts = build_col_options(df)
    label_of = {v: k for k, v in col_opts.items()}
    all_labels = list(col_opts.keys())

    with st.expander(f"📊 图{idx+1}：{sheet_name}", expanded=True):

        # ── 行1：日期列 / 数据列 / Y轴模式 ──────────────────────────
        row1 = st.columns([2, 4, 2])

        with row1[0]:
            # 优先用已保存的日期列，否则自动识别
            saved_date_col = saved.get("date_col")
            saved_date_label = label_of.get(saved_date_col) if saved_date_col else None
            auto_date_label = saved_date_label or label_of.get(auto_date, all_labels[0])
            date_label = st.selectbox(
                "日期列",
                all_labels,
                index=all_labels.index(auto_date_label) if auto_date_label in all_labels else 0,
                key=f"date_{sheet_name}",
            )
            date_col = col_opts[date_label]

        with row1[1]:
            non_date_labels = [l for l in all_labels if l != date_label]
            # 优先用已保存的数据列
            saved_val_cols = [s["col"] for s in saved.get("series", [])]
            saved_val_labels = [label_of[c] for c in saved_val_cols if c in label_of and label_of[c] in non_date_labels]
            auto_val_labels = [label_of[c] for c in auto_vals if c in label_of and label_of[c] != date_label]
            default_labels = saved_val_labels if saved_val_labels else (auto_val_labels if auto_val_labels else non_date_labels[:1])
            value_labels = st.multiselect(
                "数据列（每列一条线）",
                non_date_labels,
                default=default_labels,
                key=f"vals_{sheet_name}",
            )
            value_cols = [col_opts[l] for l in value_labels]

        with row1[2]:
            saved_ymode = saved.get("y_mode", "percent")
            y_mode_idx = 0 if saved_ymode == "percent" else 1
            y_mode = st.radio(
                "Y轴模式",
                ["percent", "points"],
                format_func=lambda x: "涨跌幅（%）" if x == "percent" else "指数点数",
                key=f"ymode_{sheet_name}",
                index=y_mode_idx,
            )

        # ── 行2：日期区间（手动输入） ──────────────────────────────
        st.caption("日期区间（留空则使用数据中的全部日期）")
        date_row = st.columns([2, 2, 4])

        # 优先用已保存的日期，否则从数据推断
        if date_series is not None and len(date_series) > 0:
            auto_start = date_series.min().strftime("%Y-%m-%d")
            auto_end   = date_series.max().strftime("%Y-%m-%d")
        else:
            auto_start = auto_end = ""

        default_start = saved.get("date_start") or auto_start
        default_end   = saved.get("date_end")   or auto_end

        with date_row[0]:
            date_start_str = st.text_input(
                "起始日期",
                value=default_start,
                placeholder="如 2020-01-01",
                key=f"ds_{sheet_name}",
            )
        with date_row[1]:
            date_end_str = st.text_input(
                "结束日期",
                value=default_end,
                placeholder="如 2024-12-31",
                key=f"de_{sheet_name}",
            )

        # ── 行3：线条名称 & 颜色 ───────────────────────────────────
        if value_cols:
            st.caption("线条设置（显示名称 / 颜色，不改颜色则使用默认）")
            sub_cols = st.columns(min(len(value_cols), 4))
            series_cfg = []
            for vi, vcol in enumerate(value_cols):
                default_color = AUTO_COLORS[vi % len(AUTO_COLORS)]
                with sub_cols[vi % len(sub_cols)]:
                    # 从已保存配置中找该列的历史名称和颜色
                    saved_series = {s["col"]: s for s in saved.get("series", [])}
                    saved_s = saved_series.get(vcol, {})
                    saved_name  = saved_s.get("name",  vcol)
                    saved_color = saved_s.get("color", default_color)

                    st.markdown(f"**线条 {vi+1}**　`{vcol}`")
                    name = st.text_input(
                        "图例名称",
                        value=saved_name,
                        key=f"name_{sheet_name}_{vi}",
                        help="图例中显示的名称",
                    )
                    # 若历史颜色不在预设列表中，回退到默认色
                    safe_saved_color = saved_color if saved_color in AUTO_COLORS else default_color
                    color = st.selectbox(
                        "线条颜色",
                        options=AUTO_COLORS,
                        format_func=lambda c: COLOR_NAMES.get(c, c),
                        index=AUTO_COLORS.index(safe_saved_color),
                        key=f"color_{sheet_name}_{vi}",
                    )
                    # 所有预设色色块一览
                    swatches = "".join(
                        f'<span title="{COLOR_NAMES.get(c, c)}" style="'
                        f'display:inline-block;width:22px;height:22px;'
                        f'background:{c};border-radius:4px;margin-right:4px;'
                        f'border:{"3px solid #333" if c == color else "2px solid #ccc"};'
                        f'"></span>'
                        for c in AUTO_COLORS
                    )
                    st.markdown(
                        f'<div style="margin-top:-6px;margin-bottom:4px">{swatches}</div>',
                        unsafe_allow_html=True,
                    )
                    series_cfg.append({"col": vcol, "name": name, "color": color})
        else:
            series_cfg = []
            st.warning("请选择至少一列数据。")

        # ── 行4：数据来源（可编辑） ────────────────────────────────
        data_source = st.text_input(
            "数据来源",
            value=saved.get("data_source", "数据来源：Wind"),
            key=f"src_{sheet_name}",
            help="默认「数据来源：Wind」，可追加截止日期等信息，如：数据来源：Wind，数据截止2026年5月",
        )

        cfg_now = {
            "date_col":    date_col,
            "series":      series_cfg,
            "y_mode":      y_mode,
            "date_start":  date_start_str.strip(),
            "date_end":    date_end_str.strip(),
            "data_source": data_source.strip(),
        }
        all_configs[sheet_name] = cfg_now

        # ── 单张生成按钮 ───────────────────────────────────────────────
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

        # 显示该张图的预览（持久化到 session_state）
        if f"preview_{sheet_name}" in st.session_state:
            _b = st.session_state[f"preview_{sheet_name}"]
            st.image(_b, use_container_width=True)
            st.download_button(
                f"⬇️ 下载 {sheet_name}.png",
                data=_b,
                file_name=f"{sheet_name}.png",
                mime="image/png",
                key=f"dl1_{sheet_name}",
                use_container_width=True,
            )

st.session_state["all_configs"] = all_configs

# ── 生成按钮 ────────────────────────────────────────────────────────
st.markdown("---")
section_title("3", "批量生成全部图表")
generate_btn = st.button(
    f"🚀 一键生成全部图表（共 {len(selected_sheets)} 张）",
    type="primary",
    use_container_width=True,
)

# ── 生成 & 展示 ─────────────────────────────────────────────────────
if generate_btn:
    results = []
    progress = st.progress(0, text="生成中...")
    total = len(selected_sheets)
    done = 0

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

        # 打包下载
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

        # 逐张展示
        cols_per_row = 2
        for i in range(0, len(results), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, r in enumerate(results[i:i+cols_per_row]):
                with row_cols[j]:
                    st.image(r["bytes"], caption=r["name"], use_container_width=True)
                    st.download_button(
                        f"⬇️ {r['name']}",
                        data=r["bytes"],
                        file_name=r["name"],
                        mime="image/png",
                        key=f"dl_{r['name']}_{i}_{j}",
                        use_container_width=True,
                    )
