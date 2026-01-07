import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus


# ==========================================
# 0. 时间与小工具
# ==========================================
CN_TZ = timezone(timedelta(hours=8))


def now_cn() -> datetime:
    """北京时间（UTC+8）"""
    return datetime.now(CN_TZ)


def is_finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def clamp(x, lo, hi) -> float:
    """对 NaN 做防护：NaN -> lo（避免评分被错误夹成满分）"""
    if not is_finite(x):
        return float(lo)
    return float(max(lo, min(hi, float(x))))


def smooth(low_score, high_score, x, x_low, x_high) -> float:
    """区间线性插值（连续化）"""
    x = float(x)
    if x <= x_low:
        return float(low_score)
    if x >= x_high:
        return float(high_score)
    ratio = (x - x_low) / (x_high - x_low)
    return float(low_score + ratio * (high_score - low_score))


def percentile_rank(x, series) -> float | None:
    """x 在 series 中的分位（0~100），忽略 NaN。"""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    return float(100.0 * (np.sum(s <= float(x)) / s.size))


def stooq_daily(symbol: str) -> pd.DataFrame | None:
    """
    从 Stooq 下载日线（避免 Yahoo 访问受限时整个应用不可用）
    返回列：Open/High/Low/Close/Volume（若有），索引为 DatetimeIndex（升序）。
    """
    try:
        s = quote_plus(symbol.lower())
        url = f"https://stooq.com/q/d/l/?s={s}&i=d"
        df = pd.read_csv(url)
        if df is None or df.empty or "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        need = {"Open", "High", "Low", "Close"}
        if not need.issubset(set(df.columns)):
            return None
        return df
    except Exception:
        return None


def pick_last_close(candidates: list[str]) -> tuple[float | None, str | None]:
    """从一组 stooq 符号里挑一个能用的 Close 最新值。"""
    for sym in candidates:
        df = stooq_daily(sym)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        val = df["Close"].dropna()
        if val.empty:
            continue
        last = float(val.iloc[-1])
        if is_finite(last):
            return last, sym
    return None, None


# ==========================================
# 1. 页面配置与 CSS
# ==========================================
st.set_page_config(page_title="纳斯达克量化决策台", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    header {visibility: hidden;}

    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        transition: transform 0.2s;
        height: 100%;
    }
    .metric-card:hover { border-color: #475569; }

    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    .status-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        float: right;
    }
    .bg-green { background-color: rgba(16, 185, 129, 0.2); color: #34d399; }
    .bg-yellow { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .bg-red { background-color: rgba(244, 63, 94, 0.2); color: #fb7185; }

    .progress-bg {
        background-color: #334155;
        height: 6px;
        border-radius: 3px;
        margin-top: 10px;
        overflow: hidden;
    }

    .rec-card {
        padding: 24px;
        border-radius: 16px;
        border: 1px solid;
        margin-bottom: 20px;
    }
    .rec-success { background: rgba(16, 185, 129, 0.1); border-color: #10b981; color: #34d399; }
    .rec-info { background: rgba(59, 130, 246, 0.1); border-color: #3b82f6; color: #60a5fa; }
    .rec-warning { background: rgba(245, 158, 11, 0.1); border-color: #f59e0b; color: #fbbf24; }
    .rec-error { background: rgba(244, 63, 94, 0.1); border-color: #f43f5e; color: #fb7185; }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. 数据获取（yfinance + Stooq 兜底 + 缺失防护）
# ==========================================
NEUTRAL = {
    "vix": 20.0,
    "us10y": 4.2,
    "dxy": 103.0,
    "pe": 30.0,
}


@st.cache_data(ttl=600)
def get_market_data():
    """
    返回 dict:
      price, ma20, ma60, rsi, drawdown, pe, vix, us10y, dxy, history,
      data_source,
      missing_vix, missing_us10y, missing_dxy
    """
    # 先尝试 Yahoo（yfinance）
    try:
        ndx = yf.Ticker("^NDX")
        hist = ndx.history(period="1y")
        if hist is None or hist.empty:
            raise RuntimeError("Empty NDX history")

        close = hist["Close"]
        current_price = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        rolling_max = close.rolling(252, min_periods=1).max()
        drawdown = float(abs((close / rolling_max - 1.0).iloc[-1] * 100))

        qqq = yf.Ticker("QQQ")
        pe = qqq.info.get("trailingPE", NEUTRAL["pe"])
        pe = float(pe) if is_finite(pe) else float(NEUTRAL["pe"])

        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        us10y = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")["Close"].iloc[-1]

        missing_vix = not is_finite(vix)
        missing_us10y = not is_finite(us10y)
        missing_dxy = not is_finite(dxy)

        vix = float(vix) if is_finite(vix) else float(NEUTRAL["vix"])
        us10y = float(us10y) if is_finite(us10y) else float(NEUTRAL["us10y"])
        dxy = float(dxy) if is_finite(dxy) else float(NEUTRAL["dxy"])

        return {
            "price": current_price,
            "ma20": ma20,
            "ma60": ma60,
            "rsi": rsi,
            "drawdown": drawdown,
            "pe": pe,
            "vix": vix,
            "us10y": us10y,
            "dxy": dxy,
            "history": hist,
            "data_source": "yfinance",
            "missing_vix": missing_vix,
            "missing_us10y": missing_us10y,
            "missing_dxy": missing_dxy,
        }
    except Exception:
        # Stooq 兜底（NDX）
        try:
            ndx_df = stooq_daily("^ndx")
            if ndx_df is None or ndx_df.empty:
                return None

            hist = ndx_df.tail(365).copy()
            close = hist["Close"]

            current_price = float(close.iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma60 = float(close.rolling(60).mean().iloc[-1])

            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])

            rolling_max = close.rolling(252, min_periods=1).max()
            drawdown = float(abs((close / rolling_max - 1.0).iloc[-1] * 100))

            vix_val, _ = pick_last_close(["vix", "^vix", "vi.f", "vx.f", "vxx.us"])
            tnx_val, _ = pick_last_close(["10yusy", "10yusy.b", "us10y", "^tnx", "tnx"])
            dxy_val, _ = pick_last_close(["dxy", "dx.f", "dx", "usdidx", "uup.us"])

            missing_vix = vix_val is None
            missing_us10y = tnx_val is None
            missing_dxy = dxy_val is None

            vix = float(vix_val) if vix_val is not None else float(NEUTRAL["vix"])
            us10y = float(tnx_val) if tnx_val is not None else float(NEUTRAL["us10y"])
            dxy = float(dxy_val) if dxy_val is not None else float(NEUTRAL["dxy"])

            pe = float(NEUTRAL["pe"])

            plot_hist = pd.DataFrame({"Close": close})
            plot_hist.index = plot_hist.index.tz_localize(None)

            return {
                "price": current_price,
                "ma20": ma20,
                "ma60": ma60,
                "rsi": rsi,
                "drawdown": drawdown,
                "pe": pe,
                "vix": vix,
                "us10y": us10y,
                "dxy": dxy,
                "history": plot_hist,
                "data_source": "stooq",
                "missing_vix": missing_vix,
                "missing_us10y": missing_us10y,
                "missing_dxy": missing_dxy,
            }
        except Exception:
            return None


# ==========================================
# 3. 连续化打分（总分<=100）
# ==========================================
def calculate_score(data):
    """
    连续化小数打分（总分<=100，硬 cap）
    各因子最大分：
      PE 25, Trend 20, DD 20, RSI 7, VIX 8, Bond 10, DXY 10 -> 合计 100
    """
    scores = {}

    p = float(data["price"])
    ma20 = float(data["ma20"])
    ma60 = float(data["ma60"])
    rsi = float(data["rsi"])
    d = float(data["drawdown"])
    v = float(data["vix"])
    u = float(data["us10y"])
    dx = float(data["dxy"])
    pe = float(data["pe"])

    miss_vix = bool(data.get("missing_vix", False))
    miss_us10y = bool(data.get("missing_us10y", False))
    miss_dxy = bool(data.get("missing_dxy", False))

    # 1) PE
    if pe < 22:
        pe_score = smooth(22, 25, pe, 15, 22)
        scores["pe"] = (pe_score, "极低估", "bg-green", "#34d399")
    elif pe < 25:
        pe_score = smooth(20, 22, pe, 22, 25)
        scores["pe"] = (pe_score, "低估", "bg-green", "#34d399")
    elif pe < 28:
        pe_score = smooth(15, 20, pe, 25, 28)
        scores["pe"] = (pe_score, "合理", "bg-yellow", "#fbbf24")
    elif pe < 32:
        pe_score = smooth(10, 15, pe, 28, 32)
        scores["pe"] = (pe_score, "偏高", "bg-yellow", "#fbbf24")
    else:
        pe_score = smooth(5, 10, pe, 32, 45)
        scores["pe"] = (pe_score, "高估", "bg-red", "#fb7185")

    # regime
    if ma20 > ma60 * 1.01:
        regime = "bull"
    elif ma20 < ma60 * 0.99:
        regime = "bear"
    else:
        regime = "range"

    dist20 = (p - ma20) / ma20 * 100.0
    dist60 = (p - ma60) / ma60 * 100.0

    # 2) Trend
    if regime == "bull":
        if p >= ma20:
            trend_score = smooth(18, 12, dist20, 0, 5)
            status, bg = "强势上行", "bg-yellow"
        elif p >= ma60:
            trend_score = smooth(16, 20, -dist20, 0, 5)
            status, bg = "上升回调", "bg-green"
        else:
            trend_score = smooth(12, 4, -dist60, 0, 10)
            status, bg = "跌破均线", "bg-red"
    elif regime == "bear":
        if p >= ma20:
            trend_score = smooth(8, 10, dist20, 0, 5)
            status, bg = "空头反弹", "bg-yellow"
        elif p >= ma60:
            trend_score = smooth(4, 8, dist60, 0, 5)
            status, bg = "弱反弹", "bg-yellow"
        else:
            trend_score = smooth(4, 0, -dist20, 0, 10)
            status, bg = "下跌延续", "bg-red"
    else:
        base = 14.0 - abs(dist20) * 2.0
        base += 2.0 if dist20 < 0 else -1.0
        trend_score = clamp(base, 0, 20)
        status = "震荡区间"
        bg = "bg-yellow" if abs(dist20) < 2 else ("bg-green" if dist20 < 0 else "bg-yellow")

    scores["trend"] = (clamp(trend_score, 0, 20), status, bg, "#34d399")

    # 3) DD
    dd_score = smooth(0, 20, d, 0, 25)
    dd_status = "回撤偏深" if d >= 15 else ("中度回撤" if d >= 8 else ("轻微回撤" if d > 0 else "新高附近"))
    dd_bg = "bg-green" if d >= 8 else ("bg-yellow" if d > 0 else "bg-red")
    scores["dd"] = (clamp(dd_score, 0, 20), dd_status, dd_bg, "#34d399")

    # 4) RSI
    if rsi < 30:
        rsi_score = smooth(7, 5.5, rsi, 10, 30)
        rsi_status, rsi_bg = "超卖", "bg-green"
    elif rsi <= 50:
        rsi_score = smooth(5.5, 4.5, rsi, 30, 50)
        rsi_status, rsi_bg = "偏弱", "bg-green"
    elif rsi <= 70:
        rsi_score = smooth(4.5, 2.5, rsi, 50, 70)
        rsi_status, rsi_bg = "偏强", "bg-yellow"
    else:
        rsi_score = smooth(2.5, 0.0, rsi, 70, 90)
        rsi_status, rsi_bg = "超买", "bg-red"
    scores["rsi"] = (clamp(rsi_score, 0, 7), rsi_status, rsi_bg, "#34d399")

    # 5) VIX
    if miss_vix:
        vix_score = 4.0
        vix_status, vix_bg = "缺失（按中性）", "bg-yellow"
    else:
        if v < 12:
            vix_score = smooth(7, 8, 12 - v, 0, 6)
            vix_status, vix_bg = "低波动", "bg-green"
        elif v < 20:
            vix_score = smooth(8, 5, v, 12, 20)
            vix_status, vix_bg = "正常波动", "bg-green"
        elif v < 28:
            vix_score = smooth(5, 2, v, 20, 28)
            vix_status, vix_bg = "波动加大", "bg-yellow"
        else:
            vix_score = smooth(2, 0, v, 28, 45)
            vix_status, vix_bg = "恐慌区", "bg-red"
    scores["vix"] = (clamp(vix_score, 0, 8), vix_status, vix_bg, "#34d399")

    # 6) Bond
    if miss_us10y:
        bond_score = 6.5
        bond_status, bond_bg = "缺失（按中性）", "bg-yellow"
    else:
        if u <= 3.5:
            bond_score, bond_status, bond_bg = 10.0, "利率友好", "bg-green"
        elif u <= 4.2:
            bond_score = smooth(10, 7.5, u, 3.5, 4.2)
            bond_status, bond_bg = "中性利率", "bg-yellow"
        elif u <= 4.8:
            bond_score = smooth(7.5, 4.5, u, 4.2, 4.8)
            bond_status, bond_bg = "偏高利率", "bg-yellow"
        else:
            bond_score = smooth(4.5, 0.0, u, 4.8, 6.0)
            bond_status, bond_bg = "高利率压制", "bg-red"
    scores["bond"] = (clamp(bond_score, 0, 10), bond_status, bond_bg, "#34d399")

    # 7) DXY
    if miss_dxy:
        dxy_score = 6.5
        dxy_status, dxy_bg = "缺失（按中性）", "bg-yellow"
    else:
        if dx <= 100:
            dxy_score, dxy_status, dxy_bg = 10.0, "弱美元", "bg-green"
        elif dx <= 104:
            dxy_score = smooth(10, 7.5, dx, 100, 104)
            dxy_status, dxy_bg = "中性美元", "bg-yellow"
        elif dx <= 106:
            dxy_score = smooth(7.5, 4.0, dx, 104, 106)
            dxy_status, dxy_bg = "强美元", "bg-yellow"
        else:
            dxy_score = smooth(4.0, 0.0, dx, 106, 112)
            dxy_status, dxy_bg = "极强美元", "bg-red"
    scores["dxy"] = (clamp(dxy_score, 0, 10), dxy_status, dxy_bg, "#34d399")

    total = float(sum(item[0] for item in scores.values()))
    total = clamp(total, 0, 100)
    return scores, total


# ==========================================
# 4. 历史分位（yfinance + Stooq 兜底；宏观缺失用中性填充）
# ==========================================
def _score_total_from_row(p, ma20, ma60, rsi, d, v, u, dx, pe):
    fake = {
        "price": p, "ma20": ma20, "ma60": ma60, "rsi": rsi, "drawdown": d,
        "vix": v, "us10y": u, "dxy": dx, "pe": pe,
        "missing_vix": False, "missing_us10y": False, "missing_dxy": False,
    }
    _, total = calculate_score(fake)
    return total


@st.cache_data(ttl=24 * 3600)
def get_calibration_series(pe_for_history: float):
    pe_const = float(pe_for_history) if is_finite(pe_for_history) else float(NEUTRAL["pe"])

    try:
        ndx = yf.download("^NDX", period="5y", interval="1d", progress=False)
        if ndx is None or ndx.empty:
            raise RuntimeError("ndx empty")

        vix = yf.download("^VIX", period="5y", interval="1d", progress=False)
        tnx = yf.download("^TNX", period="5y", interval="1d", progress=False)
        dxy = yf.download("DX-Y.NYB", period="5y", interval="1d", progress=False)

        df = pd.DataFrame(index=ndx.index)
        df["ndx"] = ndx["Close"]
        df["vix"] = (vix["Close"].reindex(df.index).ffill() if (vix is not None and not vix.empty) else NEUTRAL["vix"])
        df["us10y"] = (tnx["Close"].reindex(df.index).ffill() if (tnx is not None and not tnx.empty) else NEUTRAL["us10y"])
        df["dxy"] = (dxy["Close"].reindex(df.index).ffill() if (dxy is not None and not dxy.empty) else NEUTRAL["dxy"])

    except Exception:
        ndx_df = stooq_daily("^ndx")
        if ndx_df is None or ndx_df.empty:
            return None

        df = ndx_df.tail(1260).copy()
        df = df.rename(columns={"Close": "ndx"})
        df["vix"] = float(NEUTRAL["vix"])
        df["us10y"] = float(NEUTRAL["us10y"])
        df["dxy"] = float(NEUTRAL["dxy"])

    df["ma20"] = df["ndx"].rolling(20).mean()
    df["ma60"] = df["ndx"].rolling(60).mean()

    delta = df["ndx"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    rolling_max = df["ndx"].rolling(252, min_periods=1).max()
    df["drawdown"] = ((df["ndx"] / rolling_max - 1.0).abs() * 100)

    def _row_score(row):
        cols = ["ndx", "ma20", "ma60", "rsi", "drawdown", "vix", "us10y", "dxy"]
        if not all(is_finite(row[c]) for c in cols):
            return np.nan
        return _score_total_from_row(
            p=float(row["ndx"]),
            ma20=float(row["ma20"]),
            ma60=float(row["ma60"]),
            rsi=float(row["rsi"]),
            d=float(row["drawdown"]),
            v=float(row["vix"]),
            u=float(row["us10y"]),
            dx=float(row["dxy"]),
            pe=pe_const,
        )

    df["total_score"] = df.apply(_row_score, axis=1)
    series = df["total_score"].dropna()
    if series.empty:
        return None
    return {"score_series": series}


# ==========================================
# 5. 倍率与建议文案（0 / 1.0 / 1.25 / 1.5 / 1.75 / 2.0）
# ==========================================
def decide_mult_by_pct(pct_5y: float | None, total_raw: float, macro_hard_risk: bool) -> float:
    if pct_5y is not None:
        if pct_5y >= 90:
            mult = 2.0
        elif pct_5y >= 80:
            mult = 1.75
        elif pct_5y >= 70:
            mult = 1.5
        elif pct_5y >= 60:
            mult = 1.25
        elif pct_5y >= 40:
            mult = 1.0
        else:
            mult = 0.0
    else:
        if total_raw >= 90:
            mult = 2.0
        elif total_raw >= 80:
            mult = 1.75
        elif total_raw >= 70:
            mult = 1.5
        elif total_raw >= 60:
            mult = 1.25
        elif total_raw >= 40:
            mult = 1.0
        else:
            mult = 0.0

    if macro_hard_risk:
        mult = min(mult, 1.0)

    return float(mult)


def mult_to_label(mult: float) -> str:
    if mult <= 0:
        return "停止投（0x）"
    return f"{mult:.2f}x".rstrip("0").rstrip(".")


def rec_style(mult: float) -> tuple[str, str]:
    if mult >= 2.0:
        return "rec-success", "🚀"
    if mult >= 1.5:
        return "rec-info", "👌"
    if mult >= 1.0:
        return "rec-info", "👌"
    return "rec-warning", "⚠️"


def build_ac_text(mult: float) -> tuple[str, str]:
    if mult <= 0:
        a = (
            "A类：停止投（0x）<br>"
            "• 本周不新增，底仓继续持有。<br>"
            "• 等回到“正常/加投”区间再恢复。"
        )
        c = (
            "C类：停止投（0x）<br>"
            "• 本周不新增，避免在高风险阶段加码。<br>"
            "• 若你用 C 做短期，等信号转回再开。"
        )
        return a, c

    if mult <= 1.0:
        a = (
            "A类：正常投（1.0x）<br>"
            "• 维持每周基准金额即可。<br>"
            "• 重点是长期复利，不需要追涨杀跌。"
        )
        c = (
            "C类：正常投（1.0x）<br>"
            "• 只做小额补充即可。<br>"
            "• 如果你会做波段，记得控制持有周期。"
        )
        return a, c

    if mult <= 1.25:
        a = (
            "A类：加投（1.25x）<br>"
            "• 处在偏便宜/偏安全的区间。<br>"
            "• 比基准多投 25%，用来提高长期成本优势。"
        )
        c = (
            "C类：加投（1.25x）<br>"
            "• 可以小幅加速，但别把它当长期主仓。<br>"
            "• 更适合短期“加一点”，然后回归正常投。"
        )
        return a, c

    if mult <= 1.5:
        a = (
            "A类：加投（1.5x）<br>"
            "• 属于明显更便宜的区间。<br>"
            "• 建议优先加在 A 类，拿更长周期。"
        )
        c = (
            "C类：加投（1.5x）<br>"
            "• 适合短期加速，但要有退出纪律。<br>"
            "• 目标是“加速买入”，不是“长期持有”。"
        )
        return a, c

    if mult <= 1.75:
        a = (
            "A类：重仓加投（1.75x）<br>"
            "• 处在非常有性价比的位置。<br>"
            "• 把当周额度拉高，用长期持有换更低成本。"
        )
        c = (
            "C类：重仓加投（1.75x）<br>"
            "• 只建议小仓位使用，避免长期拖着。<br>"
            "• 如果你不做波段，建议把加速主要放 A 类。"
        )
        return a, c

    a = (
        "A类：重仓加投（2.0x）<br>"
        "• 极端便宜/极端安全区间。<br>"
        "• 直接把当周额度拉到 2 倍，长期拿住。"
    )
    c = (
        "C类：重仓加投（2.0x）<br>"
        "• 仅适用于你明确要做短期加速。<br>"
        "• 如果不做波段，建议 2x 主要放在 A 类。"
    )
    return a, c


# ==========================================
# 6. UI 卡片
# ==========================================
def render_card(title, value, subtext, score_info, max_score):
    score, status, bg_class, bar_color = score_info
    pct = (float(score) / max_score) * 100

    st.markdown(
        f"""
    <div class="metric-card">
        <div style="overflow:hidden; margin-bottom:8px;">
            <span class="metric-title">{title}</span>
            <span class="status-badge {bg_class}">{status}</span>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
        <div class="progress-bg">
            <div style="width: {pct}%; height: 100%; background-color: {bar_color}; border-radius: 3px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:0.7rem; color:#64748b;">
            <span>得分</span>
            <span style="font-family:monospace; color:#94a3b8;">{float(score):.2f}/{max_score}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ==========================================
# 7. 主程序
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<h1 style="color:#34d399; margin-bottom:0;">🦅 纳斯达克 100 决策台</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; margin-top:5px;">wyh的纳斯达克看板</p>', unsafe_allow_html=True)

with col_h2:
    if st.button("🔄 刷新数据"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f"""
        <div style="text-align:right; color:#64748b; font-size:0.75rem; margin-top:5px;">
            更新时间：<span style="color:#34d399; font-weight:bold;">
            {now_cn().strftime('%Y-%m-%d %H:%M:%S')}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.spinner("正在获取数据..."):
    data = get_market_data()

if data is None:
    st.error("无法获取数据：请检查网络连接，或确认能访问 Yahoo Finance / Stooq。")
    st.stop()

scores, total_raw = calculate_score(data)

# 宏观极端环境：风险上限（不允许加速）
vix = float(data["vix"])
us10y = float(data["us10y"])
dxy = float(data["dxy"])
macro_hard_risk = (vix >= 30) or (us10y >= 5.0) or (dxy >= 107)

# 显示口径：极端宏观时视觉提醒（不改 raw 分数）
total_display = float(total_raw)
if macro_hard_risk and total_display > 55:
    total_display = 55.0

# 市场状态
p, ma20, ma60 = float(data["price"]), float(data["ma20"]), float(data["ma60"])
if ma20 > ma60 * 1.01:
    regime_text = "上升趋势"
elif ma20 < ma60 * 0.99:
    regime_text = "下降趋势"
else:
    regime_text = "震荡区间"

# 过去5年分位（尽量保证有值）
calib = get_calibration_series(pe_for_history=float(data["pe"]))
pct_5y = None if calib is None else percentile_rank(total_raw, calib["score_series"].values)
pct_text = "—" if pct_5y is None else f"{pct_5y:.0f}"

# 倍率建议
mult = decide_mult_by_pct(pct_5y=pct_5y, total_raw=total_raw, macro_hard_risk=macro_hard_risk)
rec_class, icon = rec_style(mult)

# 区间文字（新增 1.25 与 1.75）
band_text = "区间：0–40 停投｜40–60 1.0x｜60–70 1.25x｜70–80 1.5x｜80–90 1.75x｜90–100 2.0x"

# A/C 文案（更细、更短）
a_text, c_text = build_ac_text(mult)

risk_line = ""
if macro_hard_risk:
    risk_line = "⚠ 风险上限触发：本周不加速（最多 1.0x）。"

rec_title = f"{icon} 本周建议：{mult_to_label(mult)}"
rec_msg = (
    f"市场状态：{regime_text}<br>"
    f"原始总分：{total_raw:.1f} / 100；显示总分：{total_display:.1f} / 100<br>"
    f"过去5年分位：{pct_text} / 100（越高=越偏“便宜/更适合加投”）<br><br>"
    f"{a_text}<br><br>{c_text}"
)
if risk_line:
    rec_msg += f"<br><br><span style='font-size:0.9rem; opacity:0.95;'>{risk_line}</span>"

st.markdown("---")

# 第一行：总分与建议 + 走势图
col1, col2 = st.columns([1.5, 2.5])

with col1:
    st.markdown(
        f"""
        <div style="background:#1e293b; border-radius:16px; padding:20px; text-align:center; height:100%; border:1px solid #334155;">
            <div style="color:#64748b; font-size:0.9rem; letter-spacing:1px; margin-bottom:10px;">量化总分</div>
            <div style="font-size:4rem; font-weight:900; color:{'#34d399' if total_display >= 55 else '#f43f5e'}; text-shadow: 0 0 20px rgba(255,255,255,0.1);">
                {total_display:.1f}
            </div>
            <div style="color:#475569; font-size:0.8rem;">
                满分 100 ｜ {band_text}
            </div>
            <div class="{rec_class}" style="margin-top:20px; text-align:left;">
                <div style="font-weight:bold; font-size:1.1rem; margin-bottom:10px;">{rec_title}</div>
                <div style="font-size:0.95rem; opacity:0.95; line-height:1.55;">{rec_msg}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["history"].index,
            y=data["history"]["Close"],
            mode="lines",
            name="NDX",
            line=dict(color="#0ea5e9", width=2),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.1)",
        )
    )
    fig.update_layout(
        title={"text": f"NDX 纳指走势 (当前: {data['price']:.2f})", "font": {"color": "#e2e8f0"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        xaxis=dict(showgrid=False, color="#64748b"),
        yaxis=dict(showgrid=True, gridcolor="#334155", color="#64748b"),
    )
    st.plotly_chart(fig, use_container_width=True)

# 因子矩阵（你确认的布局：4 + 3）
st.markdown('<h3 style="margin-top:30px; color:#e2e8f0; font-size:1.2rem;">📊 因子分析矩阵</h3>', unsafe_allow_html=True)

r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
with r1_c1:
    render_card("市盈率 PE", f"{data['pe']:.2f}", "QQQ TTM", scores["pe"], 25)
with r1_c2:
    render_card("回撤幅度", f"-{data['drawdown']:.2f}%", "相比252日高点", scores["dd"], 20)
with r1_c3:
    render_card("RSI (14)", f"{data['rsi']:.1f}", "相对强弱指标", scores["rsi"], 7)
with r1_c4:
    vix_value = "—" if data.get("missing_vix", False) else f"{data['vix']:.2f}"
    vix_sub = "波动率" if not data.get("missing_vix", False) else "缺失（按中性）"
    render_card("恐慌指数 VIX", vix_value, vix_sub, scores["vix"], 8)

st.markdown('<div style="margin-top:15px;"></div>', unsafe_allow_html=True)
r2_c1, r2_c2, r2_c3 = st.columns(3)
with r2_c1:
    u_value = "—" if data.get("missing_us10y", False) else f"{data['us10y']:.2f}%"
    u_sub = "无风险利率" if not data.get("missing_us10y", False) else "缺失（按中性）"
    render_card("10年美债收益率", u_value, u_sub, scores["bond"], 10)
with r2_c2:
    dx_value = "—" if data.get("missing_dxy", False) else f"{data['dxy']:.2f}"
    dx_sub = "美元强度" if not data.get("missing_dxy", False) else "缺失（按中性）"
    render_card("美元指数 DXY", dx_value, dx_sub, scores["dxy"], 10)
with r2_c3:
    render_card("趋势得分", f"{scores['trend'][0]:.2f}", "MA20/MA60 位置", scores["trend"], 20)

# 底部指南（按你给的原文）
st.markdown(
    """
    <div style="margin-top:24px; padding:12px; border:1px solid #1e40af; background:rgba(30, 64, 175, 0.1);
                border-radius:8px; color:#93c5fd; font-size:0.85rem; line-height:1.6;">
        <strong>基金 A/C 类操作指南：</strong><br>
        • <strong>A类 (前端收费)</strong>：适合长期持有 (&gt;2年)，管理费通常较低。当系统提示“买入/持有”时优先考虑。<br>
        • <strong>C类 (销售服务费)</strong>：适合短期波段 (&lt;1年)，买卖灵活但持有成本随时间增加。适合“抄底”或博反弹。
    </div>
    """,
    unsafe_allow_html=True,
)
