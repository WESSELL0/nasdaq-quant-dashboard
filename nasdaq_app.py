import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

CN_TZ = timezone(timedelta(hours=8))
NEUTRAL = {"vix": 20.0, "us10y": 4.2, "dxy": 103.0, "pe": 30.0}
TROY_OUNCE_GRAMS = 31.1035
GLD_OUNCE_APPROX = 0.093


# ==========================================
# 0. General helpers
# ==========================================
def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def is_finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def safe_float(x, default: float = np.nan) -> float:
    return float(x) if is_finite(x) else float(default)


def clamp(x, lo, hi) -> float:
    if not is_finite(x):
        return float(lo)
    return float(max(lo, min(hi, float(x))))


def smooth(low_score, high_score, x, x_low, x_high) -> float:
    x = float(x)
    if x <= x_low:
        return float(low_score)
    if x >= x_high:
        return float(high_score)
    ratio = (x - x_low) / (x_high - x_low)
    return float(low_score + ratio * (high_score - low_score))


def percentile_rank(x, series) -> float | None:
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0 or not is_finite(x):
        return None
    return float(100.0 * (np.sum(s <= float(x)) / s.size))


def range_position(series) -> float:
    s = pd.Series(series).dropna()
    if s.empty:
        return float("nan")
    low = safe_float(s.min())
    high = safe_float(s.max())
    last = safe_float(s.iloc[-1])
    if not is_finite(low) or not is_finite(high) or high <= low:
        return 50.0
    return float((last - low) / (high - low) * 100.0)


def source_label(source_key: str | None) -> str:
    mapping = {"yfinance": "Yahoo Finance", "stooq": "Stooq"}
    return mapping.get(source_key or "", source_key or "未知")


def format_number(value, digits: int = 2, suffix: str = "", missing: str = "—") -> str:
    if not is_finite(value):
        return missing
    return f"{float(value):,.{digits}f}{suffix}"


def format_metric_status(
    name: str,
    value: float,
    missing: bool,
    source: str | None,
    *,
    digits: int = 2,
    suffix: str = "",
    is_proxy: bool = False,
    neutral_fallback: bool = False,
    missing_note: str = "缺失",
) -> str:
    if missing:
        return f"{name}：{missing_note}"

    notes = []
    if is_proxy:
        notes.append("代理值")
    if neutral_fallback:
        notes.append("中性值替代")
    if source:
        notes.append(source)

    note_text = f"（{'；'.join(notes)}）" if notes else ""
    return f"{name}：{format_number(value, digits=digits, suffix=suffix)}{note_text}"


def build_data_quality_summary(data: dict) -> tuple[str, str]:
    issues = []
    critical_count = 0

    if data.get("used_fallback_data_source", False):
        issues.append("主要价格使用了 fallback 数据源")
        critical_count += 1

    if data.get("missing_vix", False):
        issues.append("VIX 缺失")
        critical_count += 1
    elif data.get("vix_is_proxy", False):
        issues.append("VIX 为代理值")
        critical_count += 1

    if data.get("missing_us10y", False):
        issues.append("10Y 美债收益率缺失")
        critical_count += 1

    if data.get("missing_dxy", False):
        issues.append("DXY 缺失")
        critical_count += 1
    elif data.get("dxy_is_proxy", False):
        issues.append("DXY 为代理值")
        critical_count += 1

    if data.get("uses_pe_metric", False):
        if data.get("missing_pe", False):
            issues.append("PE 缺失")
            critical_count += 1
        elif data.get("pe_is_neutral_fallback", False):
            issues.append("PE 使用中性值替代")
            critical_count += 1

    asset_key = data.get("asset_key")
    if asset_key == "gold":
        if data.get("missing_gold_basis", False):
            issues.append("国际黄金美元价缺失")
            critical_count += 1
        if data.get("missing_usd_cny", False):
            issues.append("USD/CNY 缺失")
            critical_count += 1
        elif data.get("gold_cny_is_estimated", False):
            issues.append("人民币/克为近似换算价")
            critical_count += 1

    if asset_key == "btc" and data.get("missing_usd_cny", False):
        issues.append("BTC 人民币参考价缺少 USD/CNY")
        critical_count += 1

    if asset_key == "hstech" and data.get("missing_usd_cny", False):
        issues.append("USD/CNY 缺失")
        critical_count += 1

    if not issues:
        label = "高"
        reason = "关键指标都来自主要数据源，且没有使用中性值替代。"
    elif critical_count <= 1 and len(issues) <= 2:
        label = "中"
        reason = "；".join(issues)
    else:
        label = "低"
        reason = "；".join(issues)

    if data.get("price_is_proxy", False):
        proxy_note = data.get("price_proxy_note") or "价格使用代理展示"
        reason = f"{reason}；{proxy_note}" if reason else proxy_note

    return label, reason

def stooq_daily(symbol: str) -> pd.DataFrame | None:
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


def normalize_history(close_series: pd.Series) -> pd.DataFrame:
    hist = pd.DataFrame({"Close": close_series.copy()})
    try:
        hist.index = hist.index.tz_localize(None)
    except Exception:
        pass
    return hist


def compute_rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return safe_float(rsi.iloc[-1])


def compute_technical_snapshot(close: pd.Series) -> dict:
    close = close.dropna()
    if close.empty:
        raise RuntimeError("Empty close series")

    price = safe_float(close.iloc[-1])
    ma20 = safe_float(close.rolling(20).mean().iloc[-1])
    ma60 = safe_float(close.rolling(60).mean().iloc[-1])
    rsi = compute_rsi(close)

    rolling_max_252 = close.rolling(252, min_periods=1).max()
    drawdown_252 = abs((price / safe_float(rolling_max_252.iloc[-1]) - 1.0) * 100.0)

    high20 = safe_float(close.tail(20).max())
    high60 = safe_float(close.tail(60).max())
    drawdown20 = abs((price / high20 - 1.0) * 100.0) if is_finite(high20) and high20 > 0 else float("nan")
    drawdown60 = abs((price / high60 - 1.0) * 100.0) if is_finite(high60) and high60 > 0 else float("nan")
    position20 = range_position(close.tail(20))

    return {
        "price": price,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": rsi,
        "drawdown": safe_float(drawdown_252),
        "drawdown20": safe_float(drawdown20),
        "drawdown60": safe_float(drawdown60),
        "position20": safe_float(position20),
        "history": normalize_history(close),
    }


def infer_regime(ma20: float, ma60: float, band: float = 0.01) -> str:
    if ma20 > ma60 * (1.0 + band):
        return "bull"
    if ma20 < ma60 * (1.0 - band):
        return "bear"
    return "range"


def get_yf_last_close(ticker: str, period: str = "5d") -> tuple[float | None, str | None]:
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None, None
        val = safe_float(hist["Close"].iloc[-1])
        if not is_finite(val):
            return None, None
        return val, f"Yahoo Finance ({ticker})"
    except Exception:
        return None, None





def get_yf_history_candidate(tickers: list[str], period: str = "1y") -> tuple[pd.Series | None, str | None]:
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if hist is None or hist.empty or "Close" not in hist.columns:
                continue
            close = hist["Close"].dropna()
            if close.empty:
                continue
            return close, ticker
        except Exception:
            continue
    return None, None


def get_stooq_history_candidate(symbols: list[str], tail: int = 365) -> tuple[pd.Series | None, str | None]:
    for symbol in symbols:
        df = stooq_daily(symbol)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if tail and len(close) > tail:
            close = close.tail(tail)
        if close.empty:
            continue
        return close, symbol
    return None, None


def get_usd_cny_reference() -> tuple[float, bool, str | None]:
    usd_cny, source_label_text = get_yf_last_close("USDCNY=X")
    if is_finite(usd_cny):
        return safe_float(usd_cny), False, source_label_text

    usd_cny, stooq_symbol = pick_last_close(["usdcny"])
    if is_finite(usd_cny):
        return safe_float(usd_cny), True, f"Stooq ({stooq_symbol})" if stooq_symbol else None

    return float("nan"), True, None


def normalize_component_score(total_points: float, available_max: float) -> float:
    if available_max <= 0:
        return 0.0
    return clamp(total_points / available_max * 100.0, 0, 100)


def build_btc_reference_quote(btc_usd_price: float) -> dict:
    usd_cny, usd_cny_is_fallback, usd_cny_source_label = get_usd_cny_reference()
    btc_cny_price = float(btc_usd_price * usd_cny) if is_finite(btc_usd_price) and is_finite(usd_cny) else float("nan")
    missing_usd_cny = not is_finite(usd_cny)
    missing_btc_cny_price = not is_finite(btc_cny_price)

    if missing_btc_cny_price:
        quality_label = "低"
        quality_reason = "缺少 USD/CNY，当前无法稳定换算 BTC 人民币参考价。"
    elif usd_cny_is_fallback:
        quality_label = "中"
        quality_reason = "BTC 人民币参考价可用，但 USD/CNY 来自 fallback 数据源。"
    else:
        quality_label = "高"
        quality_reason = "BTC 人民币参考价由 BTC 美元价与主数据源 USD/CNY 直接换算。"

    return {
        "usd_cny": safe_float(usd_cny),
        "usd_cny_source_label": usd_cny_source_label,
        "usd_cny_is_fallback": usd_cny_is_fallback,
        "missing_usd_cny": missing_usd_cny,
        "btc_cny_price": safe_float(btc_cny_price),
        "missing_btc_cny_price": missing_btc_cny_price,
        "btc_cny_quality_label": quality_label,
        "btc_cny_quality_reason": quality_reason,
        "btc_cny_formula_label": "BTC 美元价 × USD/CNY",
    }


def get_hsi_reference_snapshot() -> dict:
    close, ticker = get_yf_history_candidate(["^HSI"], period="1y")
    source_label_text = f"Yahoo Finance ({ticker})" if ticker else None

    if close is None:
        close, symbol = get_stooq_history_candidate(["^hsi"], tail=365)
        source_label_text = f"Stooq ({symbol})" if symbol else None

    if close is None:
        return {
            "missing_hsi": True,
            "hsi_level": float("nan"),
            "hsi_regime_key": None,
            "hsi_regime_text": "缺失",
            "hsi_source_label": None,
        }

    snapshot = compute_technical_snapshot(close)
    regime_key = infer_regime(snapshot["ma20"], snapshot["ma60"], band=0.01)
    regime_text = {"bull": "上升趋势", "bear": "下降趋势", "range": "震荡区间"}[regime_key]
    return {
        "missing_hsi": False,
        "hsi_level": safe_float(snapshot["price"]),
        "hsi_regime_key": regime_key,
        "hsi_regime_text": regime_text,
        "hsi_source_label": source_label_text,
    }

def gold_cny_per_gram_from_usd_oz(gold_usd_oz: float, usd_cny: float) -> float:
    if not is_finite(gold_usd_oz) or not is_finite(usd_cny):
        return float("nan")
    return float(gold_usd_oz * usd_cny / TROY_OUNCE_GRAMS)


def derive_gold_usd_oz_from_gld(gld_price: float) -> float:
    if not is_finite(gld_price) or GLD_OUNCE_APPROX <= 0:
        return float("nan")
    return float(gld_price / GLD_OUNCE_APPROX)


def build_gold_reference_quote(gld_price: float, gld_ma20: float) -> dict:
    gold_usd_oz, gold_basis_source_label = get_yf_last_close("GC=F")
    gold_basis_kind = "国际黄金美元价（GC=F）"
    gold_cny_is_estimated = False

    if not is_finite(gold_usd_oz):
        gold_usd_oz, stooq_symbol = pick_last_close(["xauusd"])
        gold_basis_source_label = f"Stooq ({stooq_symbol})" if stooq_symbol else None
        gold_basis_kind = "国际黄金美元价（XAUUSD）"

    if not is_finite(gold_usd_oz):
        gold_usd_oz = derive_gold_usd_oz_from_gld(gld_price)
        if is_finite(gold_usd_oz):
            gold_basis_source_label = f"GLD approximate ({GLD_OUNCE_APPROX:.3f} oz/share)"
            gold_basis_kind = "由 GLD 近似反推的国际黄金美元价"
            gold_cny_is_estimated = True

    usd_cny, usd_cny_source_label = get_yf_last_close("USDCNY=X")
    if not is_finite(usd_cny):
        usd_cny, stooq_symbol = pick_last_close(["usdcny"])
        usd_cny_source_label = f"Stooq ({stooq_symbol})" if stooq_symbol else None

    gold_cny_per_gram = gold_cny_per_gram_from_usd_oz(gold_usd_oz, usd_cny)
    gold_cny_ma20 = float("nan")
    gold_cny_vs_ma20_pct = float("nan")
    if is_finite(gold_cny_per_gram) and is_finite(gld_price) and gld_price > 0 and is_finite(gld_ma20) and gld_ma20 > 0:
        gold_cny_ma20 = float(gold_cny_per_gram * gld_ma20 / gld_price)
        gold_cny_vs_ma20_pct = float((gold_cny_per_gram / gold_cny_ma20 - 1.0) * 100.0) if gold_cny_ma20 > 0 else float("nan")

    missing_gold_basis = not is_finite(gold_usd_oz)
    missing_usd_cny = not is_finite(usd_cny)
    missing_gold_cny_per_gram = not is_finite(gold_cny_per_gram)

    if missing_gold_cny_per_gram:
        gold_conversion_quality_label = "低"
        gold_conversion_quality_reason = "缺少国际金价或 USD/CNY，当前无法稳定换算人民币/克参考价。"
    elif gold_cny_is_estimated:
        gold_conversion_quality_label = "低"
        gold_conversion_quality_reason = "当前人民币/克价格由 GLD 近似反推，不是精确现货报价。"
    elif (gold_basis_source_label or "").startswith("Stooq") or (usd_cny_source_label or "").startswith("Stooq"):
        gold_conversion_quality_label = "中"
        gold_conversion_quality_reason = "换算值完整，但至少一项来自 fallback 数据源。"
    else:
        gold_conversion_quality_label = "高"
        gold_conversion_quality_reason = "国际金价和 USD/CNY 都来自主要数据源，可作为较直接的投资参考。"

    if gold_basis_source_label and usd_cny_source_label:
        gold_cny_source_label = f"{gold_basis_source_label} + {usd_cny_source_label}"
    elif gold_basis_source_label:
        gold_cny_source_label = gold_basis_source_label
    else:
        gold_cny_source_label = usd_cny_source_label

    if missing_gold_cny_per_gram:
        gold_cny_display_note = "人民币/克参考价当前缺失，可能是国际金价或 USD/CNY 暂时不可用。"
    elif gold_cny_is_estimated:
        gold_cny_display_note = "基于 GLD 趋势代理与 USD/CNY 近似换算，仅作投资参考。"
    else:
        gold_cny_display_note = "基于国际黄金价格与 USD/CNY 汇率换算，仅作投资参考。"

    return {
        "gold_usd_oz": safe_float(gold_usd_oz),
        "gold_basis_kind": gold_basis_kind,
        "gold_basis_source_label": gold_basis_source_label,
        "missing_gold_basis": missing_gold_basis,
        "usd_cny": safe_float(usd_cny),
        "usd_cny_source_label": usd_cny_source_label,
        "missing_usd_cny": missing_usd_cny,
        "gold_cny_per_gram": safe_float(gold_cny_per_gram),
        "gold_cny_source_label": gold_cny_source_label,
        "missing_gold_cny_per_gram": missing_gold_cny_per_gram,
        "gold_cny_is_estimated": gold_cny_is_estimated,
        "gold_cny_formula_label": "国际金价（美元/盎司） × USD/CNY ÷ 31.1035",
        "gold_cny_display_note": gold_cny_display_note,
        "gold_cny_ma20": safe_float(gold_cny_ma20),
        "gold_cny_vs_ma20_pct": safe_float(gold_cny_vs_ma20_pct),
        "gold_conversion_quality_label": gold_conversion_quality_label,
        "gold_conversion_quality_reason": gold_conversion_quality_reason,
    }


# ==========================================
# 1. Page config and shared styles
# ==========================================
st.set_page_config(page_title="多资产投资决策台", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    header { visibility: hidden; }

    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        height: 100%;
    }

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
        line-height: 1.1;
    }

    .metric-sub {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 6px;
        line-height: 1.45;
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

    .equal-box {
        height: 560px;
        border-radius: 16px;
        border: 1px solid #334155;
        background: #1e293b;
        padding: 18px;
        box-sizing: border-box;
    }
    .summary-kicker {
        color: #64748b;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin-bottom: 6px;
        text-align: center;
    }

    .summary-score {
        text-align: center;
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1;
    }

    .summary-sub {
        color: #475569;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 10px;
    }

    .rec-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid;
        margin-top: 16px;
        line-height: 1.6;
    }

    .rec-scroll {
        height: 315px;
        overflow: auto;
        padding-right: 6px;
    }

    .rec-success { background: rgba(16, 185, 129, 0.10); border-color: #10b981; color: #34d399; }
    .rec-info { background: rgba(59, 130, 246, 0.10); border-color: #3b82f6; color: #60a5fa; }
    .rec-warning { background: rgba(245, 158, 11, 0.10); border-color: #f59e0b; color: #fbbf24; }
    .rec-error { background: rgba(244, 63, 94, 0.10); border-color: #f43f5e; color: #fb7185; }

    .status-box, .explain-box {
        background-color: #111827;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-top: 18px;
    }

    .box-title {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .status-list, .explain-list {
        margin: 0;
        padding-left: 18px;
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.92rem;
    }

    .status-note {
        color: #93c5fd;
        font-size: 0.85rem;
        margin-top: 10px;
        line-height: 1.6;
    }

    .section-title {
        margin-top: 24px;
        color: #e2e8f0;
        font-size: 1.2rem;
    }

    [data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 700;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. Data fetching
# ==========================================
@st.cache_data(ttl=600)
def get_nasdaq_market_data() -> dict | None:
    try:
        ndx = yf.Ticker("^NDX")
        hist = ndx.history(period="1y")
        if hist is None or hist.empty:
            raise RuntimeError("Empty NDX history")

        snapshot = compute_technical_snapshot(hist["Close"])
        pe_raw = yf.Ticker("QQQ").info.get("trailingPE")
        vix_raw = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        us10y_raw = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
        dxy_raw = yf.Ticker("DX-Y.NYB").history(period="5d")["Close"].iloc[-1]

        missing_pe = not is_finite(pe_raw)
        missing_vix = not is_finite(vix_raw)
        missing_us10y = not is_finite(us10y_raw)
        missing_dxy = not is_finite(dxy_raw)

        return {
            **snapshot,
            "asset_key": "nasdaq",
            "asset_label": "纳斯达克 100",
            "history_label": "NDX",
            "price_label": "纳斯达克 100 指数",
            "data_source": "yfinance",
            "used_fallback_data_source": False,
            "price_is_proxy": False,
            "proxy_expected": False,
            "price_proxy_note": None,
            "price_source_label": "Yahoo Finance (^NDX)",
            "vix": safe_float(vix_raw),
            "us10y": safe_float(us10y_raw),
            "dxy": safe_float(dxy_raw),
            "pe": safe_float(pe_raw),
            "uses_pe_metric": True,
            "missing_vix": missing_vix,
            "missing_us10y": missing_us10y,
            "missing_dxy": missing_dxy,
            "missing_pe": missing_pe,
            "vix_is_proxy": False,
            "dxy_is_proxy": False,
            "pe_is_neutral_fallback": False,
            "vix_source_label": "Yahoo Finance (^VIX)" if not missing_vix else None,
            "us10y_source_label": "Yahoo Finance (^TNX)" if not missing_us10y else None,
            "dxy_source_label": "Yahoo Finance (DX-Y.NYB)" if not missing_dxy else None,
            "pe_source_label": "Yahoo Finance (QQQ trailingPE)" if not missing_pe else None,
        }
    except Exception:
        try:
            ndx_df = stooq_daily("^ndx")
            if ndx_df is None or ndx_df.empty:
                return None

            snapshot = compute_technical_snapshot(ndx_df.tail(365)["Close"])
            vix_val, vix_symbol = pick_last_close(["vix"])
            tnx_val, tnx_symbol = pick_last_close(["10yusy", "us10y", "^tnx", "tnx"])
            dxy_val, dxy_symbol = pick_last_close(["dxy"])

            missing_vix = vix_val is None
            missing_us10y = tnx_val is None
            missing_dxy = dxy_val is None

            return {
                **snapshot,
                "asset_key": "nasdaq",
                "asset_label": "纳斯达克 100",
                "history_label": "NDX",
                "price_label": "纳斯达克 100 指数",
                "data_source": "stooq",
                "used_fallback_data_source": True,
                "price_is_proxy": False,
                "proxy_expected": False,
                "price_proxy_note": None,
                "price_source_label": "Stooq (^ndx)",
                "vix": safe_float(vix_val),
                "us10y": safe_float(tnx_val),
                "dxy": safe_float(dxy_val),
                "pe": float("nan"),
                "uses_pe_metric": True,
                "missing_vix": missing_vix,
                "missing_us10y": missing_us10y,
                "missing_dxy": missing_dxy,
                "missing_pe": True,
                "vix_is_proxy": False,
                "dxy_is_proxy": False,
                "pe_is_neutral_fallback": False,
                "vix_source_label": f"Stooq ({vix_symbol})" if vix_symbol else None,
                "us10y_source_label": f"Stooq ({tnx_symbol})" if tnx_symbol else None,
                "dxy_source_label": f"Stooq ({dxy_symbol})" if dxy_symbol else None,
                "pe_source_label": None,
            }
        except Exception:
            return None


@st.cache_data(ttl=600)
def get_gold_market_data() -> dict | None:
    try:
        gld = yf.Ticker("GLD")
        hist = gld.history(period="1y")
        if hist is None or hist.empty:
            raise RuntimeError("Empty GLD history")

        snapshot = compute_technical_snapshot(hist["Close"])
        reference_quote = build_gold_reference_quote(snapshot["price"], snapshot["ma20"])
        vix_raw = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        us10y_raw = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
        dxy_raw = yf.Ticker("DX-Y.NYB").history(period="5d")["Close"].iloc[-1]

        missing_vix = not is_finite(vix_raw)
        missing_us10y = not is_finite(us10y_raw)
        missing_dxy = not is_finite(dxy_raw)

        return {
            **snapshot,
            **reference_quote,
            "asset_key": "gold",
            "asset_label": "黄金",
            "history_label": "GLD",
            "price_label": "GLD 趋势代理价格",
            "data_source": "yfinance",
            "used_fallback_data_source": False,
            "price_is_proxy": True,
            "proxy_expected": True,
            "price_proxy_note": "趋势分析使用 GLD ETF 代理，不等同于现货金。",
            "price_source_label": "Yahoo Finance (GLD)",
            "vix": safe_float(vix_raw),
            "us10y": safe_float(us10y_raw),
            "dxy": safe_float(dxy_raw),
            "pe": float("nan"),
            "uses_pe_metric": False,
            "missing_vix": missing_vix,
            "missing_us10y": missing_us10y,
            "missing_dxy": missing_dxy,
            "missing_pe": False,
            "vix_is_proxy": False,
            "dxy_is_proxy": False,
            "pe_is_neutral_fallback": False,
            "vix_source_label": "Yahoo Finance (^VIX)" if not missing_vix else None,
            "us10y_source_label": "Yahoo Finance (^TNX)" if not missing_us10y else None,
            "dxy_source_label": "Yahoo Finance (DX-Y.NYB)" if not missing_dxy else None,
            "pe_source_label": None,
        }
    except Exception:
        try:
            gld_df = stooq_daily("gld.us")
            if gld_df is None or gld_df.empty:
                gld_df = stooq_daily("gld")
            if gld_df is None or gld_df.empty:
                return None

            snapshot = compute_technical_snapshot(gld_df.tail(365)["Close"])
            reference_quote = build_gold_reference_quote(snapshot["price"], snapshot["ma20"])
            vix_val, vix_symbol = pick_last_close(["vix"])
            tnx_val, tnx_symbol = pick_last_close(["10yusy", "us10y", "^tnx", "tnx"])
            dxy_val, dxy_symbol = pick_last_close(["dxy"])

            missing_vix = vix_val is None
            missing_us10y = tnx_val is None
            missing_dxy = dxy_val is None

            return {
                **snapshot,
                **reference_quote,
                "asset_key": "gold",
                "asset_label": "黄金",
                "history_label": "GLD",
                "price_label": "GLD 趋势代理价格",
                "data_source": "stooq",
                "used_fallback_data_source": True,
                "price_is_proxy": True,
                "proxy_expected": True,
                "price_proxy_note": "趋势分析使用 GLD ETF 代理，不等同于现货金。",
                "price_source_label": "Stooq (gld.us)",
                "vix": safe_float(vix_val),
                "us10y": safe_float(tnx_val),
                "dxy": safe_float(dxy_val),
                "pe": float("nan"),
                "uses_pe_metric": False,
                "missing_vix": missing_vix,
                "missing_us10y": missing_us10y,
                "missing_dxy": missing_dxy,
                "missing_pe": False,
                "vix_is_proxy": False,
                "dxy_is_proxy": False,
                "pe_is_neutral_fallback": False,
                "vix_source_label": f"Stooq ({vix_symbol})" if vix_symbol else None,
                "us10y_source_label": f"Stooq ({tnx_symbol})" if tnx_symbol else None,
                "dxy_source_label": f"Stooq ({dxy_symbol})" if dxy_symbol else None,
                "pe_source_label": None,
            }
        except Exception:
            return None



@st.cache_data(ttl=600)
def get_btc_market_data() -> dict | None:
    try:
        close, ticker = get_yf_history_candidate(["BTC-USD"])
        if close is None:
            raise RuntimeError("Empty BTC history")

        snapshot = compute_technical_snapshot(close)
        reference_quote = build_btc_reference_quote(snapshot["price"])
        price_source_label = f"Yahoo Finance ({ticker})"
        vix_raw = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        us10y_raw = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
        dxy_raw = yf.Ticker("DX-Y.NYB").history(period="5d")["Close"].iloc[-1]

        missing_vix = not is_finite(vix_raw)
        missing_us10y = not is_finite(us10y_raw)
        missing_dxy = not is_finite(dxy_raw)

        return {
            **snapshot,
            **reference_quote,
            "asset_key": "btc",
            "asset_label": "比特币",
            "history_label": "BTC",
            "price_label": "BTC 美元参考价",
            "data_source": "yfinance",
            "used_fallback_data_source": False,
            "price_is_proxy": False,
            "proxy_expected": False,
            "price_proxy_note": None,
            "price_source_label": price_source_label,
            "btc_cny_source_label": f"{price_source_label} + {reference_quote.get('usd_cny_source_label')}" if reference_quote.get("usd_cny_source_label") else None,
            "vix": safe_float(vix_raw),
            "us10y": safe_float(us10y_raw),
            "dxy": safe_float(dxy_raw),
            "pe": float("nan"),
            "uses_pe_metric": False,
            "missing_vix": missing_vix,
            "missing_us10y": missing_us10y,
            "missing_dxy": missing_dxy,
            "missing_pe": False,
            "vix_is_proxy": False,
            "dxy_is_proxy": False,
            "pe_is_neutral_fallback": False,
            "vix_source_label": "Yahoo Finance (^VIX)" if not missing_vix else None,
            "us10y_source_label": "Yahoo Finance (^TNX)" if not missing_us10y else None,
            "dxy_source_label": "Yahoo Finance (DX-Y.NYB)" if not missing_dxy else None,
            "pe_source_label": None,
        }
    except Exception:
        try:
            close, symbol = get_stooq_history_candidate(["btcusd"])
            if close is None:
                return None

            snapshot = compute_technical_snapshot(close)
            reference_quote = build_btc_reference_quote(snapshot["price"])
            price_source_label = f"Stooq ({symbol})"
            vix_val, vix_symbol = pick_last_close(["vix"])
            tnx_val, tnx_symbol = pick_last_close(["10yusy", "us10y", "^tnx", "tnx"])
            dxy_val, dxy_symbol = pick_last_close(["dxy"])

            missing_vix = vix_val is None
            missing_us10y = tnx_val is None
            missing_dxy = dxy_val is None

            return {
                **snapshot,
                **reference_quote,
                "asset_key": "btc",
                "asset_label": "比特币",
                "history_label": "BTC",
                "price_label": "BTC 美元参考价",
                "data_source": "stooq",
                "used_fallback_data_source": True,
                "price_is_proxy": False,
                "proxy_expected": False,
                "price_proxy_note": None,
                "price_source_label": price_source_label,
                "btc_cny_source_label": f"{price_source_label} + {reference_quote.get('usd_cny_source_label')}" if reference_quote.get("usd_cny_source_label") else None,
                "vix": safe_float(vix_val),
                "us10y": safe_float(tnx_val),
                "dxy": safe_float(dxy_val),
                "pe": float("nan"),
                "uses_pe_metric": False,
                "missing_vix": missing_vix,
                "missing_us10y": missing_us10y,
                "missing_dxy": missing_dxy,
                "missing_pe": False,
                "vix_is_proxy": False,
                "dxy_is_proxy": False,
                "pe_is_neutral_fallback": False,
                "vix_source_label": f"Stooq ({vix_symbol})" if vix_symbol else None,
                "us10y_source_label": f"Stooq ({tnx_symbol})" if tnx_symbol else None,
                "dxy_source_label": f"Stooq ({dxy_symbol})" if dxy_symbol else None,
                "pe_source_label": None,
            }
        except Exception:
            return None

def get_hstech_market_data() -> dict | None:
    yf_candidates = ["3033.HK", "3067.HK"]
    stooq_candidates = ["3033.hk", "3067.hk"]

    try:
        close, ticker = get_yf_history_candidate(yf_candidates)
        if close is None:
            raise RuntimeError("Empty HSTECH proxy history")

        snapshot = compute_technical_snapshot(close)
        hsi_reference = get_hsi_reference_snapshot()
        dxy_raw = yf.Ticker("DX-Y.NYB").history(period="5d")["Close"].iloc[-1]
        us10y_raw = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
        usd_cny, usd_cny_is_fallback, usd_cny_source_label = get_usd_cny_reference()

        missing_dxy = not is_finite(dxy_raw)
        missing_us10y = not is_finite(us10y_raw)
        missing_usd_cny = not is_finite(usd_cny)

        return {
            **snapshot,
            **hsi_reference,
            "asset_key": "hstech",
            "asset_label": "恒生科技",
            "history_label": ticker or "HSTECH proxy",
            "price_label": "恒生科技 ETF 代理价",
            "data_source": "yfinance",
            "used_fallback_data_source": False,
            "price_is_proxy": True,
            "proxy_expected": True,
            "price_proxy_note": "当前价格使用恒生科技 ETF 代理，不等同于恒生科技指数点位本体。",
            "price_source_label": f"Yahoo Finance ({ticker})",
            "proxy_symbol": ticker,
            "proxy_label": f"{ticker} ETF 代理" if ticker else "恒生科技 ETF 代理",
            "vix": float("nan"),
            "us10y": safe_float(us10y_raw),
            "dxy": safe_float(dxy_raw),
            "usd_cny": safe_float(usd_cny),
            "usd_cny_source_label": usd_cny_source_label,
            "usd_cny_is_fallback": usd_cny_is_fallback,
            "missing_usd_cny": missing_usd_cny,
            "pe": float("nan"),
            "uses_pe_metric": False,
            "missing_vix": True,
            "missing_us10y": missing_us10y,
            "missing_dxy": missing_dxy,
            "missing_pe": False,
            "vix_is_proxy": False,
            "dxy_is_proxy": False,
            "pe_is_neutral_fallback": False,
            "vix_source_label": None,
            "us10y_source_label": "Yahoo Finance (^TNX)" if not missing_us10y else None,
            "dxy_source_label": "Yahoo Finance (DX-Y.NYB)" if not missing_dxy else None,
            "pe_source_label": None,
        }
    except Exception:
        try:
            close, symbol = get_stooq_history_candidate(stooq_candidates)
            if close is None:
                return None

            snapshot = compute_technical_snapshot(close)
            hsi_reference = get_hsi_reference_snapshot()
            dxy_val, dxy_symbol = pick_last_close(["dxy"])
            tnx_val, tnx_symbol = pick_last_close(["10yusy", "us10y", "^tnx", "tnx"])
            usd_cny, usd_cny_is_fallback, usd_cny_source_label = get_usd_cny_reference()

            missing_dxy = dxy_val is None
            missing_us10y = tnx_val is None
            missing_usd_cny = not is_finite(usd_cny)

            return {
                **snapshot,
                **hsi_reference,
                "asset_key": "hstech",
                "asset_label": "恒生科技",
                "history_label": symbol.upper() if symbol else "HSTECH proxy",
                "price_label": "恒生科技 ETF 代理价",
                "data_source": "stooq",
                "used_fallback_data_source": True,
                "price_is_proxy": True,
                "proxy_expected": True,
                "price_proxy_note": "当前价格使用恒生科技 ETF 代理，不等同于恒生科技指数点位本体。",
                "price_source_label": f"Stooq ({symbol})",
                "proxy_symbol": symbol,
                "proxy_label": f"{symbol} ETF 代理" if symbol else "恒生科技 ETF 代理",
                "vix": float("nan"),
                "us10y": safe_float(tnx_val),
                "dxy": safe_float(dxy_val),
                "usd_cny": safe_float(usd_cny),
                "usd_cny_source_label": usd_cny_source_label,
                "usd_cny_is_fallback": usd_cny_is_fallback,
                "missing_usd_cny": missing_usd_cny,
                "pe": float("nan"),
                "uses_pe_metric": False,
                "missing_vix": True,
                "missing_us10y": missing_us10y,
                "missing_dxy": missing_dxy,
                "missing_pe": False,
                "vix_is_proxy": False,
                "dxy_is_proxy": False,
                "pe_is_neutral_fallback": False,
                "vix_source_label": None,
                "us10y_source_label": f"Stooq ({tnx_symbol})" if tnx_symbol else None,
                "dxy_source_label": f"Stooq ({dxy_symbol})" if dxy_symbol else None,
                "pe_source_label": None,
            }
        except Exception:
            return None


# ==========================================
# 3. Nasdaq module
# ==========================================

# ==========================================
# 3. Nasdaq module
# ==========================================
def calculate_nasdaq_score(data: dict) -> tuple[dict, float]:
    scores = {}
    total_points = 0.0
    available_max = 0.0

    p = safe_float(data.get("price"))
    ma20 = safe_float(data.get("ma20"))
    ma60 = safe_float(data.get("ma60"))
    rsi = safe_float(data.get("rsi"))
    drawdown = safe_float(data.get("drawdown"))
    vix = safe_float(data.get("vix"))
    us10y = safe_float(data.get("us10y"))
    dxy = safe_float(data.get("dxy"))
    pe = safe_float(data.get("pe"))

    miss_vix = bool(data.get("missing_vix", False))
    miss_us10y = bool(data.get("missing_us10y", False))
    miss_dxy = bool(data.get("missing_dxy", False))
    miss_pe = bool(data.get("missing_pe", not is_finite(pe)))

    if miss_pe:
        scores["pe"] = (0.0, "缺失", "bg-yellow", "#64748b")
    else:
        if pe < 22:
            pe_score = smooth(22, 25, pe, 15, 22)
            pe_status, pe_bg = "极低估", "bg-green"
        elif pe < 25:
            pe_score = smooth(20, 22, pe, 22, 25)
            pe_status, pe_bg = "低估", "bg-green"
        elif pe < 28:
            pe_score = smooth(15, 20, pe, 25, 28)
            pe_status, pe_bg = "合理", "bg-yellow"
        elif pe < 32:
            pe_score = smooth(10, 15, pe, 28, 32)
            pe_status, pe_bg = "偏高", "bg-yellow"
        else:
            pe_score = smooth(5, 10, pe, 32, 45)
            pe_status, pe_bg = "高估", "bg-red"
        pe_score = clamp(pe_score, 0, 25)
        scores["pe"] = (pe_score, pe_status, pe_bg, "#34d399")
        total_points += pe_score
        available_max += 25.0

    regime = infer_regime(ma20, ma60, band=0.01)
    dist20 = (p - ma20) / ma20 * 100.0 if is_finite(ma20) and ma20 != 0 else 0.0
    dist60 = (p - ma60) / ma60 * 100.0 if is_finite(ma60) and ma60 != 0 else 0.0

    if regime == "bull":
        if p >= ma20:
            trend_score = smooth(18, 12, dist20, 0, 5)
            trend_status, trend_bg = "强势上行", "bg-yellow"
        elif p >= ma60:
            trend_score = smooth(16, 20, -dist20, 0, 5)
            trend_status, trend_bg = "上升回调", "bg-green"
        else:
            trend_score = smooth(12, 4, -dist60, 0, 10)
            trend_status, trend_bg = "跌破均线", "bg-red"
    elif regime == "bear":
        if p >= ma20:
            trend_score = smooth(8, 10, dist20, 0, 5)
            trend_status, trend_bg = "空头反弹", "bg-yellow"
        elif p >= ma60:
            trend_score = smooth(4, 8, dist60, 0, 5)
            trend_status, trend_bg = "弱反弹", "bg-yellow"
        else:
            trend_score = smooth(4, 0, -dist20, 0, 10)
            trend_status, trend_bg = "下跌延续", "bg-red"
    else:
        base = 14.0 - abs(dist20) * 2.0
        base += 2.0 if dist20 < 0 else -1.0
        trend_score = clamp(base, 0, 20)
        trend_status = "震荡区间"
        trend_bg = "bg-yellow" if abs(dist20) < 2 else ("bg-green" if dist20 < 0 else "bg-yellow")

    trend_score = clamp(trend_score, 0, 20)
    scores["trend"] = (trend_score, trend_status, trend_bg, "#34d399")
    total_points += trend_score
    available_max += 20.0

    dd_score = clamp(smooth(0, 20, drawdown, 0, 25), 0, 20)
    if drawdown >= 15:
        dd_status, dd_bg = "回撤偏深", "bg-green"
    elif drawdown >= 8:
        dd_status, dd_bg = "中度回撤", "bg-green"
    elif drawdown > 0:
        dd_status, dd_bg = "轻微回撤", "bg-yellow"
    else:
        dd_status, dd_bg = "新高附近", "bg-red"
    scores["dd"] = (dd_score, dd_status, dd_bg, "#34d399")
    total_points += dd_score
    available_max += 20.0

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
    rsi_score = clamp(rsi_score, 0, 7)
    scores["rsi"] = (rsi_score, rsi_status, rsi_bg, "#34d399")
    total_points += rsi_score
    available_max += 7.0

    if miss_vix:
        vix_score = 4.0
        vix_status, vix_bg = "缺失", "bg-yellow"
    else:
        if vix < 12:
            vix_score = smooth(7, 8, 12 - vix, 0, 6)
            vix_status, vix_bg = "低波动", "bg-green"
        elif vix < 20:
            vix_score = smooth(8, 5, vix, 12, 20)
            vix_status, vix_bg = "正常波动", "bg-green"
        elif vix < 28:
            vix_score = smooth(5, 2, vix, 20, 28)
            vix_status, vix_bg = "波动加大", "bg-yellow"
        else:
            vix_score = smooth(2, 0, vix, 28, 45)
            vix_status, vix_bg = "恐慌区", "bg-red"
    vix_score = clamp(vix_score, 0, 8)
    scores["vix"] = (vix_score, vix_status, vix_bg, "#34d399")
    total_points += vix_score
    available_max += 8.0

    if miss_us10y:
        bond_score = 6.5
        bond_status, bond_bg = "缺失", "bg-yellow"
    else:
        if us10y <= 3.5:
            bond_score, bond_status, bond_bg = 10.0, "利率友好", "bg-green"
        elif us10y <= 4.2:
            bond_score = smooth(10, 7.5, us10y, 3.5, 4.2)
            bond_status, bond_bg = "中性利率", "bg-yellow"
        elif us10y <= 4.8:
            bond_score = smooth(7.5, 4.5, us10y, 4.2, 4.8)
            bond_status, bond_bg = "偏高利率", "bg-yellow"
        else:
            bond_score = smooth(4.5, 0.0, us10y, 4.8, 6.0)
            bond_status, bond_bg = "高利率压制", "bg-red"
    bond_score = clamp(bond_score, 0, 10)
    scores["bond"] = (bond_score, bond_status, bond_bg, "#34d399")
    total_points += bond_score
    available_max += 10.0

    if miss_dxy:
        dxy_score = 6.5
        dxy_status, dxy_bg = "缺失", "bg-yellow"
    else:
        if dxy <= 100:
            dxy_score, dxy_status, dxy_bg = 10.0, "弱美元", "bg-green"
        elif dxy <= 104:
            dxy_score = smooth(10, 7.5, dxy, 100, 104)
            dxy_status, dxy_bg = "中性美元", "bg-yellow"
        elif dxy <= 106:
            dxy_score = smooth(7.5, 4.0, dxy, 104, 106)
            dxy_status, dxy_bg = "强美元", "bg-yellow"
        else:
            dxy_score = smooth(4.0, 0.0, dxy, 106, 112)
            dxy_status, dxy_bg = "极强美元", "bg-red"
    dxy_score = clamp(dxy_score, 0, 10)
    scores["dxy"] = (dxy_score, dxy_status, dxy_bg, "#34d399")
    total_points += dxy_score
    available_max += 10.0

    total = 0.0 if available_max <= 0 else float(total_points / available_max * 100.0)
    return scores, clamp(total, 0, 100)


def _nasdaq_total_from_row(p, ma20, ma60, rsi, drawdown, vix, us10y, dxy, pe):
    fake = {
        "price": p,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": rsi,
        "drawdown": drawdown,
        "vix": vix,
        "us10y": us10y,
        "dxy": dxy,
        "pe": pe,
        "missing_vix": False,
        "missing_us10y": False,
        "missing_dxy": False,
        "missing_pe": False,
    }
    _, total = calculate_nasdaq_score(fake)
    return total


@st.cache_data(ttl=24 * 3600)
def get_nasdaq_calibration_series(pe_for_history: float | None) -> dict | None:
    pe_const = safe_float(pe_for_history, default=NEUTRAL["pe"])

    try:
        ndx = yf.download("^NDX", period="5y", interval="1d", progress=False, auto_adjust=False)
        if ndx is None or ndx.empty:
            raise RuntimeError("ndx empty")
        df = pd.DataFrame(index=ndx.index)
        df["ndx"] = ndx["Close"]
    except Exception:
        ndx_df = stooq_daily("^ndx")
        if ndx_df is None or ndx_df.empty:
            return None
        df = ndx_df.tail(1260).copy().rename(columns={"Close": "ndx"})

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
    df["drawdown"] = (df["ndx"] / rolling_max - 1.0).abs() * 100.0

    def _row_score(row):
        cols = ["ndx", "ma20", "ma60", "rsi", "drawdown", "vix", "us10y", "dxy"]
        if not all(is_finite(row[c]) for c in cols):
            return np.nan
        return _nasdaq_total_from_row(
            p=safe_float(row["ndx"]),
            ma20=safe_float(row["ma20"]),
            ma60=safe_float(row["ma60"]),
            rsi=safe_float(row["rsi"]),
            drawdown=safe_float(row["drawdown"]),
            vix=safe_float(row["vix"]),
            us10y=safe_float(row["us10y"]),
            dxy=safe_float(row["dxy"]),
            pe=pe_const,
        )

    df["total_score"] = df.apply(_row_score, axis=1)
    series = df["total_score"].dropna()
    if series.empty:
        return None
    return {"score_series": series}


def decide_mult_by_score(total_score: float) -> float:
    score = float(total_score)
    if score < 40:
        mult = 0.0
    elif score < 60:
        mult = 1.0
    elif score < 70:
        mult = 1.25
    elif score < 80:
        mult = 1.5
    elif score < 90:
        mult = 1.75
    else:
        mult = 2.0
    return float(mult)


def apply_macro_risk_cap(mult: float, macro_hard_risk: bool) -> float:
    return float(min(mult, 1.0)) if macro_hard_risk else float(mult)


def mult_to_label(mult: float) -> str:
    if mult <= 0:
        return "停止投（0x）"
    return f"{mult:.2f}x".rstrip("0").rstrip(".")


def nasdaq_action_label(mult: float) -> str:
    if mult <= 0:
        return "控制仓位 / 暂停新增"
    if mult <= 1.0:
        return "适合正常定投"
    if mult <= 1.25:
        return "适合小幅加仓"
    if mult <= 1.5:
        return "适合分批加仓"
    if mult <= 1.75:
        return "适合积极分批加仓"
    return "适合高配分批加仓"


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
        return (
            "暂停新增，本周只保留长期底仓。",
            "暂停新增，避免在高风险阶段做短线加码。",
        )
    if mult == 1.0:
        return (
            "按每周基准金额正常定投。",
            "可以小额参与，但不要把 C 类当长期主仓。",
        )
    if mult == 1.25:
        return (
            "比基准金额多投 25%，优先放在长期仓。",
            "可以小仓位加速，但仍要控制持有周期。",
        )
    if mult == 1.5:
        return (
            "提高当周额度，适合分批拉长周期布局。",
            "只适合短期加速，不建议重仓长期拿。",
        )
    if mult == 1.75:
        return (
            "明显偏向加仓，继续以长期持有思路执行。",
            "仅建议小仓位参与，并保留退出纪律。",
        )
    return (
        "属于极端便宜区附近，允许把周额度拉到 2 倍。",
        "只适合短期加速，若没有纪律不建议大仓位。",
    )

def build_nasdaq_explanations(
    data: dict,
    total_score: float,
    regime_text: str,
    pct_text: str,
    final_mult: float,
    uncapped_mult: float,
    macro_hard_risk: bool,
) -> list[str]:
    notes = []

    if data.get("missing_pe", False):
        notes.append("PE 当前缺失，本次总分更多反映技术面和宏观环境，而不是实时估值。")
    else:
        pe = safe_float(data.get("pe"))
        if pe < 25:
            notes.append("PE 处于偏低到合理区间，长期定投的赔率相对更友好。")
        elif pe >= 32:
            notes.append("PE 仍偏高，节奏上更适合分批，而不是一次性追高。")
        else:
            notes.append("PE 处于中性区间，动作优先看趋势、回撤和宏观环境。")

    if regime_text == "上升趋势":
        notes.append("均线结构仍偏上行，但如果短线离 20 日均线过远，追涨的性价比会下降。")
    elif regime_text == "下降趋势":
        notes.append("均线结构仍偏弱，只有回撤更深或宏观风险回落时，加仓节奏才更舒服。")
    else:
        notes.append("当前更像震荡区间，按分档定投会比主观择时更稳。")

    if macro_hard_risk and uncapped_mult > final_mult:
        notes.append(
            f"模型原始建议倍率是 {mult_to_label(uncapped_mult)}，但因为 VIX / 利率 / 美元触发风险上限，最终显示被压到 {mult_to_label(final_mult)}。"
        )
    else:
        notes.append(f"当前建议倍率为 {mult_to_label(final_mult)}，更适合按周节奏执行，而不是一次性重仓。")

    if pct_text != "—":
        notes.append(f"历史技术分位近似值位于 {pct_text}/100，可用来判断当前位置冷热，但它不是独立交易信号。")
    else:
        notes.append(f"当前总分为 {total_score:.1f} / 100，可先把它理解为“技术 + 宏观”的相对舒适度。")

    return notes

# ==========================================
# 4. Gold module
# ==========================================
def calculate_gold_view(data: dict) -> dict:
    env_scores = {}
    tech_scores = {}

    price = safe_float(data.get("price"))
    ma20 = safe_float(data.get("ma20"))
    ma60 = safe_float(data.get("ma60"))
    rsi = safe_float(data.get("rsi"))
    drawdown20 = safe_float(data.get("drawdown20"))
    drawdown60 = safe_float(data.get("drawdown60"))
    position20 = safe_float(data.get("position20"))
    vix = safe_float(data.get("vix"))
    us10y = safe_float(data.get("us10y"))
    dxy = safe_float(data.get("dxy"))

    env_points = 0.0
    env_max = 0.0
    tech_points = 0.0
    tech_max = 0.0

    if data.get("missing_dxy", False):
        dxy_score = 10.0
        dxy_status, dxy_bg = "缺失", "bg-yellow"
    else:
        if dxy <= 100:
            dxy_score, dxy_status, dxy_bg = 20.0, "弱美元", "bg-green"
        elif dxy <= 103:
            dxy_score = smooth(20, 14, dxy, 100, 103)
            dxy_status, dxy_bg = "美元偏弱", "bg-green"
        elif dxy <= 106:
            dxy_score = smooth(14, 8, dxy, 103, 106)
            dxy_status, dxy_bg = "美元中性", "bg-yellow"
        else:
            dxy_score = smooth(8, 0, dxy, 106, 112)
            dxy_status, dxy_bg = "美元偏强", "bg-red"
    dxy_score = clamp(dxy_score, 0, 20)
    env_scores["dxy"] = (dxy_score, dxy_status, dxy_bg, "#34d399")
    env_points += dxy_score
    env_max += 20.0

    if data.get("missing_us10y", False):
        bond_score = 9.0
        bond_status, bond_bg = "缺失", "bg-yellow"
    else:
        if us10y <= 3.8:
            bond_score, bond_status, bond_bg = 18.0, "利率友好", "bg-green"
        elif us10y <= 4.3:
            bond_score = smooth(18, 12, us10y, 3.8, 4.3)
            bond_status, bond_bg = "利率中性", "bg-yellow"
        elif us10y <= 4.8:
            bond_score = smooth(12, 6, us10y, 4.3, 4.8)
            bond_status, bond_bg = "利率偏高", "bg-yellow"
        else:
            bond_score = smooth(6, 0, us10y, 4.8, 6.0)
            bond_status, bond_bg = "高利率压制", "bg-red"
    bond_score = clamp(bond_score, 0, 18)
    env_scores["bond"] = (bond_score, bond_status, bond_bg, "#34d399")
    env_points += bond_score
    env_max += 18.0

    if data.get("missing_vix", False):
        vix_score = 6.0
        vix_status, vix_bg = "缺失", "bg-yellow"
    else:
        if vix < 12:
            vix_score = smooth(2, 4, vix, 8, 12)
            vix_status, vix_bg = "风险情绪平静", "bg-yellow"
        elif vix < 20:
            vix_score = smooth(4, 7, vix, 12, 20)
            vix_status, vix_bg = "风险情绪温和", "bg-yellow"
        elif vix < 30:
            vix_score = smooth(7, 12, vix, 20, 30)
            vix_status, vix_bg = "避险需求升温", "bg-green"
        else:
            vix_score = smooth(12, 10, vix, 30, 45)
            vix_status, vix_bg = "避险需求较强", "bg-green"
    vix_score = clamp(vix_score, 0, 12)
    env_scores["vix"] = (vix_score, vix_status, vix_bg, "#34d399")
    env_points += vix_score
    env_max += 12.0

    regime = infer_regime(ma20, ma60, band=0.005)
    dist20 = (price - ma20) / ma20 * 100.0 if is_finite(ma20) and ma20 != 0 else 0.0
    dist60 = (price - ma60) / ma60 * 100.0 if is_finite(ma60) and ma60 != 0 else 0.0

    if regime == "bull":
        if price >= ma20 and dist20 <= 3:
            trend_score = smooth(15, 10, dist20, 0, 3)
            trend_status, trend_bg = "上行但不过热", "bg-green"
        elif price >= ma20:
            trend_score = smooth(10, 5, dist20, 3, 8)
            trend_status, trend_bg = "趋势上行偏热", "bg-yellow"
        elif price >= ma60:
            trend_score = smooth(12, 15, -dist20, 0, 4)
            trend_status, trend_bg = "上升回踩", "bg-green"
        else:
            trend_score = smooth(8, 3, -dist60, 0, 8)
            trend_status, trend_bg = "跌破关键均线", "bg-red"
    elif regime == "bear":
        if price >= ma20:
            trend_score = smooth(6, 8, dist20, 0, 3)
            trend_status, trend_bg = "弱势反弹", "bg-yellow"
        else:
            trend_score = smooth(4, 0, -dist20, 0, 8)
            trend_status, trend_bg = "下行整理", "bg-red"
    else:
        base = 10.0 - abs(dist20) * 1.5
        base += 2.0 if dist20 < 0 else -1.0
        trend_score = clamp(base, 0, 15)
        trend_status = "区间波动"
        trend_bg = "bg-green" if dist20 < 0 else "bg-yellow"
    trend_score = clamp(trend_score, 0, 15)
    tech_scores["trend"] = (trend_score, trend_status, trend_bg, "#34d399")
    tech_points += trend_score
    tech_max += 15.0

    if rsi < 35:
        rsi_score = smooth(12, 10, rsi, 20, 35)
        rsi_status, rsi_bg = "偏冷", "bg-green"
    elif rsi <= 55:
        rsi_score = smooth(10, 8, rsi, 35, 55)
        rsi_status, rsi_bg = "中性", "bg-yellow"
    elif rsi <= 68:
        rsi_score = smooth(8, 4, rsi, 55, 68)
        rsi_status, rsi_bg = "偏热", "bg-yellow"
    else:
        rsi_score = smooth(4, 0, rsi, 68, 85)
        rsi_status, rsi_bg = "过热", "bg-red"
    rsi_score = clamp(rsi_score, 0, 12)
    tech_scores["rsi"] = (rsi_score, rsi_status, rsi_bg, "#34d399")
    tech_points += rsi_score
    tech_max += 12.0

    pullback = max(safe_float(drawdown20, default=0.0), safe_float(drawdown60, default=0.0))
    if pullback < 1:
        pullback_score = 2.0
        pullback_status, pullback_bg = "接近短期高位", "bg-red"
    elif pullback < 3:
        pullback_score = smooth(2, 7, pullback, 1, 3)
        pullback_status, pullback_bg = "小幅回调", "bg-yellow"
    elif pullback < 6:
        pullback_score = smooth(7, 10, pullback, 3, 6)
        pullback_status, pullback_bg = "回调低吸区", "bg-green"
    elif pullback < 10:
        pullback_score = smooth(10, 4, pullback, 6, 10)
        pullback_status, pullback_bg = "回撤偏深", "bg-yellow"
    else:
        pullback_score = smooth(4, 0, pullback, 10, 18)
        pullback_status, pullback_bg = "趋势转弱", "bg-red"
    pullback_score = clamp(pullback_score, 0, 10)
    tech_scores["pullback"] = (pullback_score, pullback_status, pullback_bg, "#34d399")
    tech_points += pullback_score
    tech_max += 10.0

    if position20 <= 20:
        pos_score = 8.0
        pos_status, pos_bg = "20日低位", "bg-green"
    elif position20 <= 40:
        pos_score = smooth(8, 6, position20, 20, 40)
        pos_status, pos_bg = "20日偏低", "bg-green"
    elif position20 <= 60:
        pos_score = smooth(6, 4, position20, 40, 60)
        pos_status, pos_bg = "20日中位", "bg-yellow"
    elif position20 <= 80:
        pos_score = smooth(4, 1, position20, 60, 80)
        pos_status, pos_bg = "20日偏高", "bg-yellow"
    else:
        pos_score = smooth(1, 0, position20, 80, 100)
        pos_status, pos_bg = "20日高位", "bg-red"
    pos_score = clamp(pos_score, 0, 8)
    tech_scores["position"] = (pos_score, pos_status, pos_bg, "#34d399")
    tech_points += pos_score
    tech_max += 8.0

    environment_score = 0.0 if env_max <= 0 else env_points / env_max * 100.0
    technical_score = 0.0 if tech_max <= 0 else tech_points / tech_max * 100.0
    composite_score = clamp(environment_score * 0.55 + technical_score * 0.45, 0, 100)

    if environment_score >= 67:
        environment_label = "偏多"
    elif environment_score >= 45:
        environment_label = "中性"
    else:
        environment_label = "偏谨慎"

    overbought = rsi >= 70 or position20 >= 85
    pullback_buy = pullback >= 2 and position20 <= 45
    trend_supportive = regime == "bull" and price >= ma60

    if overbought:
        position_label = "高位偏热区"
    elif pullback_buy and environment_score >= 60:
        position_label = "回调低吸区"
    elif regime == "bull":
        position_label = "趋势上行区"
    elif regime == "range":
        position_label = "区间波动区"
    else:
        position_label = "整理偏弱区"

    if overbought and environment_score < 55:
        action_label = "适合逢高减仓"
    elif overbought:
        action_label = "不建议追高"
    elif environment_score >= 65 and pullback_buy:
        action_label = "适合分批低吸"
    elif environment_score >= 65 and technical_score >= 60:
        action_label = "适合买入"
    elif environment_score >= 55 and trend_supportive:
        action_label = "适合持有"
    elif environment_score < 45 and technical_score < 45:
        action_label = "适合观望"
    elif environment_score < 50 and position20 >= 70:
        action_label = "适合观望"
    else:
        action_label = "适合持有" if composite_score >= 55 else "适合观望"

    if regime == "range" and 30 <= position20 <= 75 and 40 <= rsi <= 60:
        t_label = "适合"
    elif 20 <= position20 <= 80 and 35 <= rsi <= 68 and regime != "bull":
        t_label = "一般"
    else:
        t_label = "不适合"

    explanations = []
    if data.get("missing_dxy", False):
        explanations.append("DXY 缺失，美元因子的解释力下降。")
    elif dxy <= 103:
        explanations.append("美元偏弱，对黄金环境偏友好。")
    else:
        explanations.append("美元偏强，对黄金形成一定压制。")

    if data.get("missing_us10y", False):
        explanations.append("10Y 美债收益率缺失，利率因子只能保守解读。")
    elif us10y <= 4.3:
        explanations.append("长端利率回落，黄金的持有成本压力相对更小。")
    else:
        explanations.append("长端利率偏高，会压制黄金弹性，动作上更要看价格位置。")

    if overbought:
        explanations.append("GLD 短线位置偏热，更适合持有或等回踩，不建议追高。")
    elif pullback_buy and environment_score >= 60:
        explanations.append("价格已经有一定回调，若分批布局会比追涨更从容。")
    elif regime == "range":
        explanations.append("当前位置更像区间波动，适合耐心等更舒服的出手位置。")
    else:
        explanations.append("趋势仍在，但当前位置并不是明显低位，动作上更强调节奏控制。")

    gold_cny_per_gram = safe_float(data.get("gold_cny_per_gram"))
    gold_cny_vs_ma20_pct = safe_float(data.get("gold_cny_vs_ma20_pct"))
    if is_finite(gold_cny_per_gram):
        if is_finite(gold_cny_vs_ma20_pct):
            if gold_cny_vs_ma20_pct >= 0:
                explanations.append(
                    f"当前人民币参考价约 {format_number(gold_cny_per_gram, digits=1)} 元/克，按 GLD 趋势折算后高于 20 日均线 {abs(gold_cny_vs_ma20_pct):.1f}%。"
                )
            else:
                explanations.append(
                    f"当前人民币参考价约 {format_number(gold_cny_per_gram, digits=1)} 元/克，按 GLD 趋势折算后低于 20 日均线 {abs(gold_cny_vs_ma20_pct):.1f}%。"
                )
        else:
            explanations.append(
                f"当前人民币参考价约 {format_number(gold_cny_per_gram, digits=1)} 元/克，可用来和自己的元/克交易直觉做对照。"
            )
    else:
        explanations.append("当前无法稳定换算人民币/克参考价，短线判断仍以趋势和位置为主。")

    if data.get("gold_cny_is_estimated", False):
        explanations.append("当前元/克价格由 GLD 近似反推，不等同于实时现货成交价。")

    explanations.append(f"做T建议：{t_label}。")

    return {
        "environment_scores": env_scores,
        "technical_scores": tech_scores,
        "environment_score": clamp(environment_score, 0, 100),
        "technical_score": clamp(technical_score, 0, 100),
        "composite_score": composite_score,
        "environment_label": environment_label,
        "position_label": position_label,
        "action_label": action_label,
        "t_label": t_label,
        "regime": regime,
        "pullback": pullback,
        "explanations": explanations,
    }


def gold_rec_style(action_label: str) -> tuple[str, str]:
    if action_label in {"适合买入", "适合分批低吸"}:
        return "rec-success", "🥇"
    if action_label == "适合持有":
        return "rec-info", "👌"
    if action_label in {"适合观望", "不建议追高"}:
        return "rec-warning", "⚠️"
    return "rec-error", "⛳"

# ==========================================
# 5. Shared UI helpers
# ==========================================



def action_rec_style(action_label: str, default_icon: str) -> tuple[str, str]:
    if action_label in {"适合买入", "适合分批低吸", "适合分批布局"}:
        return "rec-success", default_icon
    if action_label == "适合持有":
        return "rec-info", "👌"
    if action_label == "适合逢高减仓":
        return "rec-error", "⛳"
    if action_label in {"适合观望", "不建议追高", "适合控制仓位", "不适合抄底"}:
        return "rec-warning", "⚠️"
    return "rec-info", default_icon

def render_card(title: str, value: str, subtext: str, score_info: tuple, max_score: float) -> None:
    score, status, bg_class, bar_color = score_info
    pct = 0.0 if max_score <= 0 else clamp(float(score) / max_score * 100.0, 0, 100)
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
                <div style="width:{pct}%; height:100%; background-color:{bar_color}; border-radius:3px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:0.7rem; color:#64748b;">
                <span>得分</span>
                <span style="font-family:monospace; color:#94a3b8;">{float(score):.2f}/{max_score}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, value: str, subtext: str, status: str, bg_class: str = "bg-yellow") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="overflow:hidden; margin-bottom:8px;">
                <span class="metric-title">{title}</span>
                <span class="status-badge {bg_class}">{status}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )





def summary_score_color(score: float) -> str:
    if score >= 60:
        return "#34d399"
    if score >= 45:
        return "#fbbf24"
    return "#f43f5e"


def render_asset_summary_box(
    score_kicker: str,
    score: float,
    action_label: str,
    environment_label: str,
    price_line: str,
    brief_line: str,
    detail_lines: list[str],
    rec_class: str,
    icon: str,
    *,
    score_color: str | None = None,
) -> None:
    color = score_color or summary_score_color(score)
    details_html = "<br>".join(detail_lines)
    st.markdown(
        f"""
        <div class="equal-box" style="height:auto; min-height:550px;">
            <div class="summary-kicker">{score_kicker}</div>
            <div class="summary-score" style="color:{color};">{score:.1f}<span style="font-size:1.5rem;"> 分</span></div>
            <div class="summary-sub">当前建议：{action_label} ｜ 当前环境：{environment_label}</div>
            <div style="margin-top:12px; text-align:center; color:#f1f5f9; font-size:1rem; font-weight:700;">{price_line}</div>
            <div style="margin-top:6px; text-align:center; color:#94a3b8; font-size:0.88rem; line-height:1.55;">{brief_line}</div>
            <div class="rec-card {rec_class} rec-scroll" style="height:220px; margin-top:18px;">
                <div style="font-weight:bold; font-size:1.05rem; margin-bottom:10px;">{icon} 当前更偏向：{action_label}</div>
                <div style="font-size:0.95rem; opacity:0.95;">{details_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_chart_section(
    data: dict,
    chart_title: str,
    score_kicker: str,
    score: float,
    action_label: str,
    environment_label: str,
    price_line: str,
    brief_line: str,
    detail_lines: list[str],
    rec_class: str,
    icon: str,
    *,
    score_color: str | None = None,
) -> None:
    col1, col2 = st.columns([1.5, 2.5])
    with col1:
        render_asset_summary_box(
            score_kicker,
            score,
            action_label,
            environment_label,
            price_line,
            brief_line,
            detail_lines,
            rec_class,
            icon,
            score_color=score_color,
        )
    with col2:
        render_price_chart(data, chart_title)


def render_data_status_section(section_title: str, status_lines: list[str], quality_label: str, quality_reason: str) -> None:
    st.markdown(f"<h3 class='section-title'>{section_title}</h3>", unsafe_allow_html=True)
    render_status_box("数据状态 / 可信度", status_lines, quality_label, quality_reason)

def render_status_box(title: str, lines: list[str], quality_label: str, quality_reason: str) -> None:
    badge_class = {"高": "bg-green", "中": "bg-yellow", "低": "bg-red"}.get(quality_label, "bg-yellow")
    items = "".join(f"<li>{line}</li>" for line in lines)
    st.markdown(
        f"""
        <div class="status-box">
            <div style="overflow:hidden; margin-bottom:8px;">
                <span class="box-title">{title}</span>
                <span class="status-badge {badge_class}">可信度 {quality_label}</span>
            </div>
            <ul class="status-list">{items}</ul>
            <div class="status-note">说明：{quality_reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_box(title: str, lines: list[str]) -> None:
    items = "".join(f"<li>{line}</li>" for line in lines)
    st.markdown(
        f"""
        <div class="explain-box">
            <div class="box-title">{title}</div>
            <ul class="explain-list">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_price_chart(data: dict, title_text: str) -> None:
    close = data["history"]["Close"]
    ma20_line = close.rolling(20).mean()
    ma60_line = close.rolling(60).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["history"].index,
            y=close,
            mode="lines",
            name=data["history_label"],
            line=dict(color="#0ea5e9", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["history"].index,
            y=ma20_line,
            mode="lines",
            name="MA20",
            line=dict(color="#22c55e", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["history"].index,
            y=ma60_line,
            mode="lines",
            name="MA60",
            line=dict(color="#f59e0b", width=1, dash="dot"),
        )
    )

    fig.update_layout(
        title={"text": title_text, "font": {"color": "#e2e8f0"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=560,
        xaxis=dict(showgrid=False, color="#64748b"),
        yaxis=dict(showgrid=True, gridcolor="#334155", color="#64748b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")


def build_nasdaq_status_lines(data: dict, quality_label: str, quality_reason: str) -> list[str]:
    fallback_text = "是" if data.get("used_fallback_data_source", False) else "否"
    return [
        f"主要数据源：{source_label(data.get('data_source'))}",
        f"是否使用 fallback：{fallback_text}",
        f"{data.get('price_label')}：{format_number(data.get('price'))}（{data.get('price_source_label')}）",
        format_metric_status(
            "VIX",
            data.get("vix"),
            data.get("missing_vix", False),
            data.get("vix_source_label"),
            missing_note="缺失（未使用代理值）",
        ),
        format_metric_status(
            "DXY",
            data.get("dxy"),
            data.get("missing_dxy", False),
            data.get("dxy_source_label"),
            missing_note="缺失（未使用代理值）",
        ),
        format_metric_status(
            "10Y 美债收益率",
            data.get("us10y"),
            data.get("missing_us10y", False),
            data.get("us10y_source_label"),
            suffix="%",
            missing_note="缺失",
        ),
        format_metric_status(
            "PE",
            data.get("pe"),
            data.get("missing_pe", False),
            data.get("pe_source_label"),
            neutral_fallback=data.get("pe_is_neutral_fallback", False),
            missing_note="缺失（未计入实时 PE 得分）",
        ),
        f"当前结果可信度：{quality_label}（{quality_reason}）",
    ]


def build_gold_status_lines(data: dict, quality_label: str, quality_reason: str) -> list[str]:
    fallback_text = "是" if data.get("used_fallback_data_source", False) else "否"
    return [
        f"主要数据源：{source_label(data.get('data_source'))}",
        f"是否使用 fallback：{fallback_text}",
        "黄金参考价格：人民币/克换算参考价",
        format_metric_status(
            "人民币/克参考价",
            data.get("gold_cny_per_gram"),
            data.get("missing_gold_cny_per_gram", False),
            data.get("gold_cny_source_label"),
            digits=1,
            suffix=" 元/克",
            missing_note="缺失（缺少国际金价或 USD/CNY）",
        ),
        format_metric_status(
            "国际黄金基础价",
            data.get("gold_usd_oz"),
            data.get("missing_gold_basis", False),
            data.get("gold_basis_source_label"),
            digits=2,
            suffix=" 美元/盎司",
            is_proxy=data.get("gold_cny_is_estimated", False),
            missing_note="缺失",
        ),
        format_metric_status(
            "USD/CNY",
            data.get("usd_cny"),
            data.get("missing_usd_cny", False),
            data.get("usd_cny_source_label"),
            digits=4,
            missing_note="缺失",
        ),
        f"换算公式：{data.get('gold_cny_formula_label')}",
        f"换算可信度：{data.get('gold_conversion_quality_label')}（{data.get('gold_conversion_quality_reason')}）",
        f"GLD 趋势代理：{format_number(data.get('price'))}（{data.get('price_source_label')}）",
        format_metric_status(
            "VIX",
            data.get("vix"),
            data.get("missing_vix", False),
            data.get("vix_source_label"),
            missing_note="缺失（未使用代理值）",
        ),
        format_metric_status(
            "DXY",
            data.get("dxy"),
            data.get("missing_dxy", False),
            data.get("dxy_source_label"),
            missing_note="缺失（未使用代理值）",
        ),
        format_metric_status(
            "10Y 美债收益率",
            data.get("us10y"),
            data.get("missing_us10y", False),
            data.get("us10y_source_label"),
            suffix="%",
            missing_note="缺失",
        ),
        f"当前结果可信度：{quality_label}（{quality_reason}）",
    ]


# ==========================================
# 6. Panel renderers
# ==========================================

# ==========================================
# 6. Panel renderers
# ==========================================
def render_nasdaq_panel() -> None:
    with st.spinner("正在获取纳斯达克 100 数据..."):
        data = get_nasdaq_market_data()

    if data is None:
        st.error("无法获取纳斯达克 100 数据：请检查网络连接，或确认能访问 Yahoo Finance / Stooq。")
        return

    scores, total_score = calculate_nasdaq_score(data)
    quality_label, quality_reason = build_data_quality_summary(data)

    vix = safe_float(data.get("vix"))
    us10y = safe_float(data.get("us10y"))
    dxy = safe_float(data.get("dxy"))
    macro_hard_risk = (
        (is_finite(vix) and vix >= 30)
        or (is_finite(us10y) and us10y >= 5.0)
        or (is_finite(dxy) and dxy >= 107)
    )

    regime_key = infer_regime(safe_float(data.get("ma20")), safe_float(data.get("ma60")), band=0.01)
    regime_text = {"bull": "上升趋势", "bear": "下降趋势", "range": "震荡区间"}[regime_key]

    calib_pe = data.get("pe") if is_finite(data.get("pe")) else None
    calib = get_nasdaq_calibration_series(pe_for_history=calib_pe)
    pct_5y = None if calib is None else percentile_rank(total_score, calib["score_series"].values)
    pct_text = "—" if pct_5y is None else f"{pct_5y:.0f}"

    uncapped_mult = decide_mult_by_score(total_score)
    final_mult = apply_macro_risk_cap(uncapped_mult, macro_hard_risk)
    action_label = nasdaq_action_label(final_mult)
    rec_class, icon = rec_style(final_mult)
    a_text, c_text = build_ac_text(final_mult)

    risk_line = ""
    if macro_hard_risk and uncapped_mult > final_mult:
        risk_line = (
            f"风控提示：模型原始倍率 {mult_to_label(uncapped_mult)} 高于当前风险上限，"
            f"最终已下调为 {mult_to_label(final_mult)}。"
        )

    explanation_lines = build_nasdaq_explanations(
        data=data,
        total_score=total_score,
        regime_text=regime_text,
        pct_text=pct_text,
        final_mult=final_mult,
        uncapped_mult=uncapped_mult,
        macro_hard_risk=macro_hard_risk,
    )
    status_lines = build_nasdaq_status_lines(data, quality_label, quality_reason)

    risk_state_value = "已封顶" if macro_hard_risk and uncapped_mult > final_mult else "正常"
    risk_state_sub = (
        f"原始 {mult_to_label(uncapped_mult)} → 最终 {mult_to_label(final_mult)}"
        if macro_hard_risk and uncapped_mult > final_mult
        else "当前未触发宏观风控上限"
    )
    risk_state_bg = "bg-red" if risk_state_value == "已封顶" else "bg-green"

    if pct_5y is None:
        pct_status, pct_bg = "缺失", "bg-yellow"
    elif pct_5y <= 20:
        pct_status, pct_bg = "偏冷", "bg-green"
    elif pct_5y >= 80:
        pct_status, pct_bg = "偏热", "bg-red"
    else:
        pct_status, pct_bg = "中性", "bg-yellow"

    summary_brief = risk_line or "更适合按周节奏执行，不适合把单次信号当成一次性重仓命令。"
    summary_lines = [
        f"参考价：NDX {format_number(data.get('price'))}",
        f"建议倍率：{mult_to_label(final_mult)}",
        f"历史技术分位近似值：{pct_text} / 100",
        f"A类：{a_text}",
        f"C类：{c_text}",
    ]
    if risk_line:
        summary_lines.append(risk_line)

    st.markdown("<h3 class='section-title'>📈 纳斯达克 100 决策看板</h3>", unsafe_allow_html=True)
    render_summary_chart_section(
        data,
        f"NDX 走势（当前：{format_number(data.get('price'))}）",
        "纳指综合分",
        total_score,
        action_label,
        regime_text,
        f"参考价：NDX {format_number(data.get('price'))}",
        summary_brief,
        summary_lines,
        rec_class,
        icon,
    )

    st.markdown("<h3 class='section-title'>🧭 纳指为什么这样判断</h3>", unsafe_allow_html=True)
    render_explanation_box("判断逻辑", explanation_lines)

    st.markdown("<h3 class='section-title'>📊 纳指核心因子</h3>", unsafe_allow_html=True)
    row1 = st.columns(4)
    with row1[0]:
        render_info_card("建议倍率", mult_to_label(final_mult), "当前更适合按周节奏执行的倍投档位", "风控后", "bg-green" if final_mult >= 1.0 else "bg-yellow")
    with row1[1]:
        render_info_card("风险状态", risk_state_value, risk_state_sub, risk_state_value, risk_state_bg)
    with row1[2]:
        pe_value = "PE unavailable" if data.get("missing_pe", False) else format_number(data.get("pe"))
        pe_sub = "未计入实时 PE 得分" if data.get("missing_pe", False) else "QQQ trailing PE"
        render_card("市盈率 PE", pe_value, pe_sub, scores["pe"], 25)
    with row1[3]:
        vix_value = "—" if data.get("missing_vix", False) else format_number(data.get("vix"))
        vix_sub = "缺失（未使用代理）" if data.get("missing_vix", False) else "波动率越高越容易压低节奏"
        render_card("恐慌指数 VIX", vix_value, vix_sub, scores["vix"], 8)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    row2 = st.columns(3)
    with row2[0]:
        dxy_value = "—" if data.get("missing_dxy", False) else format_number(data.get("dxy"))
        dxy_sub = "缺失（未使用代理）" if data.get("missing_dxy", False) else "强美元通常不利于成长资产估值"
        render_card("美元指数 DXY", dxy_value, dxy_sub, scores["dxy"], 10)
    with row2[1]:
        bond_value = "—" if data.get("missing_us10y", False) else format_number(data.get("us10y"), suffix="%")
        bond_sub = "缺失" if data.get("missing_us10y", False) else "高利率会抬高成长资产压力"
        render_card("10年美债收益率", bond_value, bond_sub, scores["bond"], 10)
    with row2[2]:
        render_info_card("历史技术分位近似值", f"{pct_text} / 100", "只看当前技术/宏观得分在近 5 年的大致冷热点位置", pct_status, pct_bg)

    st.markdown("<h3 class='section-title'>🔬 纳指技术面与数据状态</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    tech_row = st.columns(4)
    with tech_row[0]:
        render_card("RSI (14)", format_number(data.get("rsi"), digits=1), "相对强弱指标", scores["rsi"], 7)
    with tech_row[1]:
        render_card("回撤幅度", f"-{format_number(data.get('drawdown'))}%", "相对 252 日高点", scores["dd"], 20)
    with tech_row[2]:
        render_card("趋势得分", format_number(scores["trend"][0]), "MA20 / MA60 位置", scores["trend"], 20)
    with tech_row[3]:
        render_info_card("当前价格位置", regime_text, "用于区分上升趋势、下降趋势或震荡区间", regime_text, "bg-green" if regime_key == "bull" else ("bg-yellow" if regime_key == "range" else "bg-red"))

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    render_data_status_section("📎 纳指数据状态", status_lines, quality_label, quality_reason)

    st.markdown(
        """
        <div style="margin-top:22px; padding:12px; border:1px solid #1e40af; background:rgba(30, 64, 175, 0.1);
                    border-radius:8px; color:#93c5fd; font-size:0.85rem; line-height:1.6;">
            <strong>使用提醒：</strong><br>
            • “历史技术分位近似值”只是把当前技术/宏观得分放到近 5 年的相对位置里观察，不是完整估值历史回放。<br>
            • 页面更适合帮你决定“定投节奏”和“是否需要放慢/加快”，不适合作为单次重仓交易命令。
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_gold_panel() -> None:
    with st.spinner("正在获取黄金数据..."):
        data = get_gold_market_data()

    if data is None:
        st.error("无法获取黄金数据：请检查网络连接，或确认能访问 Yahoo Finance / Stooq。")
        return

    view = calculate_gold_view(data)
    quality_label, quality_reason = build_data_quality_summary(data)
    status_lines = build_gold_status_lines(data, quality_label, quality_reason)
    rec_class, icon = gold_rec_style(view["action_label"])

    gold_cny_per_gram = safe_float(data.get("gold_cny_per_gram"))
    gold_usd_oz = safe_float(data.get("gold_usd_oz"))
    gold_cny_vs_ma20_pct = safe_float(data.get("gold_cny_vs_ma20_pct"))
    composite_score = safe_float(view.get("composite_score"))
    cny_price_text = "—" if not is_finite(gold_cny_per_gram) else f"¥{format_number(gold_cny_per_gram, digits=1)} / 克"
    usd_price_text = "—" if not is_finite(gold_usd_oz) else f"${format_number(gold_usd_oz, digits=2)} / 盎司"
    ma20_gap_text = "—"
    if is_finite(gold_cny_vs_ma20_pct):
        direction = "高于" if gold_cny_vs_ma20_pct >= 0 else "低于"
        ma20_gap_text = f"{direction}20日均线折算价 {abs(gold_cny_vs_ma20_pct):.1f}%"

    position_text = format_number(data.get("position20"), digits=0, suffix=" / 100")
    gold_basis_text = "—" if data.get("missing_gold_basis", False) else format_number(data.get("gold_usd_oz"), digits=2, suffix=" 美元/盎司")
    usd_cny_text = "—" if data.get("missing_usd_cny", False) else format_number(data.get("usd_cny"), digits=4)
    st.markdown("<h3 class='section-title'>🥇 黄金投资 / 做T 决策看板</h3>", unsafe_allow_html=True)
    gold_price_mode = st.radio("参考价显示", ["人民币 / 克", "美元 / 盎司"], horizontal=True, key="gold_price_mode")
    selected_price_text = cny_price_text if gold_price_mode == "人民币 / 克" else usd_price_text
    secondary_price_text = usd_price_text if gold_price_mode == "人民币 / 克" else cny_price_text
    price_line = (
        f"人民币价：{cny_price_text} ｜ 国际价：{usd_price_text}"
        if gold_price_mode == "人民币 / 克"
        else f"国际价：{usd_price_text} ｜ 人民币价：{cny_price_text}"
    )
    brief_line = f"当前更像{view['position_label']}；做T建议：{view['t_label']}。"
    if data.get("gold_cny_is_estimated", False):
        brief_line += " 当前元/克价格为近似估算值。"
    summary_lines = [
        f"人民币参考价：{cny_price_text}",
        f"国际参考价：{usd_price_text}",
        f"当前显示口径：{gold_price_mode}",
        f"动作建议：{view['action_label']}",
        f"做T建议：{view['t_label']}",
        f"距离20日均线折算价：{ma20_gap_text}",
        f"近期区间位置：20日分位 {position_text}",
        f"USD/CNY：{usd_cny_text}",
    ]

    render_summary_chart_section(
        data,
        f"GLD 走势（趋势代理，当前：{format_number(data.get('price'))} 美元）",
        "黄金综合分",
        composite_score,
        view["action_label"],
        view["environment_label"],
        price_line,
        brief_line,
        summary_lines,
        rec_class,
        icon,
    )

    st.markdown("<h3 class='section-title'>🧭 黄金为什么这样判断</h3>", unsafe_allow_html=True)
    render_explanation_box("判断逻辑", view["explanations"])

    st.markdown("<h3 class='section-title'>📊 黄金核心因子</h3>", unsafe_allow_html=True)
    row1 = st.columns(4)
    ref_status = "估算价" if data.get("gold_cny_is_estimated", False) else ("参考价" if is_finite(gold_cny_per_gram) else "缺失")
    ref_bg = "bg-yellow" if data.get("gold_cny_is_estimated", False) else ("bg-green" if is_finite(gold_cny_per_gram) else "bg-red")
    env_bg = "bg-green" if view["environment_score"] >= 60 else ("bg-yellow" if view["environment_score"] >= 45 else "bg-red")
    tech_bg = "bg-green" if view["technical_score"] >= 60 else ("bg-yellow" if view["technical_score"] >= 45 else "bg-red")
    t_bg = "bg-green" if view["t_label"] == "适合" else ("bg-yellow" if view["t_label"] == "一般" else "bg-red")
    with row1[0]:
        render_info_card("当前参考价", selected_price_text, f"当前显示 {gold_price_mode}；另一口径 {secondary_price_text}", ref_status, ref_bg)
    with row1[1]:
        render_info_card("环境分", format_number(view["environment_score"], digits=1), "越高表示美元/利率/风险情绪背景更友好", view["environment_label"], env_bg)
    with row1[2]:
        render_info_card("技术分", format_number(view["technical_score"], digits=1), "越高表示当前价格位置和节奏更舒服", view["position_label"], tech_bg)
    with row1[3]:
        render_info_card("做T建议", view["t_label"], "更适合做T、一般，还是不适合", view["t_label"], t_bg)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    row2 = st.columns(4)
    with row2[0]:
        dxy_value = "—" if data.get("missing_dxy", False) else format_number(data.get("dxy"))
        dxy_sub = "缺失（未使用代理）" if data.get("missing_dxy", False) else "弱美元更利多黄金"
        render_card("美元指数 DXY", dxy_value, dxy_sub, view["environment_scores"]["dxy"], 20)
    with row2[1]:
        bond_value = "—" if data.get("missing_us10y", False) else format_number(data.get("us10y"), suffix="%")
        bond_sub = "缺失" if data.get("missing_us10y", False) else "长端利率越低越友好"
        render_card("10年美债收益率", bond_value, bond_sub, view["environment_scores"]["bond"], 18)
    with row2[2]:
        vix_value = "—" if data.get("missing_vix", False) else format_number(data.get("vix"))
        vix_sub = "缺失（未使用代理）" if data.get("missing_vix", False) else "风险升温通常利多避险"
        render_card("VIX / 风险情绪", vix_value, vix_sub, view["environment_scores"]["vix"], 12)
    with row2[3]:
        render_info_card("USD/CNY", usd_cny_text, "人民币/克换算所用汇率", "汇率", "bg-green" if not data.get("missing_usd_cny", False) else "bg-red")

    st.markdown("<h3 class='section-title'>🔬 黄金技术面与数据状态</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    tech_row = st.columns(4)
    with tech_row[0]:
        render_card("RSI (14)", format_number(data.get("rsi"), digits=1), "短线是否过热", view["technical_scores"]["rsi"], 12)
    with tech_row[1]:
        render_card("趋势得分", format_number(view["technical_scores"]["trend"][0]), "MA20 / MA60 结构", view["technical_scores"]["trend"], 15)
    with tech_row[2]:
        pullback_text = f"20日 -{format_number(data.get('drawdown20'))}% ｜ 60日 -{format_number(data.get('drawdown60'))}%"
        render_card("回撤位置", pullback_text, "相对近 20 / 60 日高点", view["technical_scores"]["pullback"], 10)
    with tech_row[3]:
        render_card("20日价格分位", position_text, "越低越适合低吸；可与元/克参考价一起看", view["technical_scores"]["position"], 8)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    data_row = st.columns(3)
    basis_status = "近似" if data.get("gold_cny_is_estimated", False) else ("直连" if is_finite(data.get("gold_usd_oz")) else "缺失")
    basis_bg = "bg-yellow" if data.get("gold_cny_is_estimated", False) else ("bg-green" if is_finite(data.get("gold_usd_oz")) else "bg-red")
    conv_bg = "bg-green" if data.get("gold_conversion_quality_label") == "高" else ("bg-yellow" if data.get("gold_conversion_quality_label") == "中" else "bg-red")
    with data_row[0]:
        render_info_card("国际黄金基础价", gold_basis_text, data.get("gold_basis_kind") or "优先用直接国际金价，拿不到时才近似反推", basis_status, basis_bg)
    with data_row[1]:
        render_info_card("GLD 趋势代理", format_number(data.get("price"), digits=2, suffix=" 美元"), "仅作趋势 / 做T 辅助，不等同于现货金", "代理", "bg-yellow")
    with data_row[2]:
        render_info_card("换算可信度", data.get("gold_conversion_quality_label"), data.get("gold_conversion_quality_reason"), data.get("gold_conversion_quality_label"), conv_bg)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    render_data_status_section("📎 黄金数据状态", status_lines, quality_label, quality_reason)

    st.markdown(
        """
        <div style="margin-top:22px; padding:12px; border:1px solid #854d0e; background:rgba(133, 77, 14, 0.12);
                    border-radius:8px; color:#fde68a; font-size:0.85rem; line-height:1.6;">
            <strong>使用提醒：</strong><br>
            • 黄金主价格显示的是“人民币/克参考价”，基于国际金价与 USD/CNY 换算，不是金饰零售价，也不含加工费。<br>
            • GLD 仍保留为趋势代理，用于判断方向、位置和做T节奏，不等同于现货金本身。
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# 6. BTC / Hang Seng Tech modules
# ==========================================

# ==========================================
# 6. BTC / Hang Seng Tech modules
# ==========================================
def calculate_btc_view(data: dict) -> dict:
    env_scores = {}
    tech_scores = {}

    price = safe_float(data.get("price"))
    ma20 = safe_float(data.get("ma20"))
    ma60 = safe_float(data.get("ma60"))
    rsi = safe_float(data.get("rsi"))
    drawdown20 = safe_float(data.get("drawdown20"))
    drawdown60 = safe_float(data.get("drawdown60"))
    position20 = safe_float(data.get("position20"))
    vix = safe_float(data.get("vix"))
    us10y = safe_float(data.get("us10y"))
    dxy = safe_float(data.get("dxy"))

    env_points = env_max = tech_points = tech_max = 0.0

    if data.get("missing_dxy", False):
        dxy_score = 9.0
        dxy_status, dxy_bg = "缺失", "bg-yellow"
    else:
        if dxy <= 100:
            dxy_score, dxy_status, dxy_bg = 18.0, "弱美元", "bg-green"
        elif dxy <= 104:
            dxy_score = smooth(18, 12, dxy, 100, 104)
            dxy_status, dxy_bg = "中性美元", "bg-yellow"
        elif dxy <= 107:
            dxy_score = smooth(12, 5, dxy, 104, 107)
            dxy_status, dxy_bg = "强美元", "bg-yellow"
        else:
            dxy_score = smooth(5, 0, dxy, 107, 112)
            dxy_status, dxy_bg = "美元压制", "bg-red"
    dxy_score = clamp(dxy_score, 0, 18)
    env_scores["dxy"] = (dxy_score, dxy_status, dxy_bg, "#34d399")
    env_points += dxy_score; env_max += 18.0

    if data.get("missing_us10y", False):
        bond_score = 6.0
        bond_status, bond_bg = "缺失", "bg-yellow"
    else:
        if us10y <= 3.8:
            bond_score, bond_status, bond_bg = 12.0, "流动性友好", "bg-green"
        elif us10y <= 4.3:
            bond_score = smooth(12, 8.5, us10y, 3.8, 4.3)
            bond_status, bond_bg = "利率中性", "bg-yellow"
        elif us10y <= 4.9:
            bond_score = smooth(8.5, 4.0, us10y, 4.3, 4.9)
            bond_status, bond_bg = "利率偏高", "bg-yellow"
        else:
            bond_score = smooth(4.0, 0.0, us10y, 4.9, 6.0)
            bond_status, bond_bg = "利率压制", "bg-red"
    bond_score = clamp(bond_score, 0, 12)
    env_scores["bond"] = (bond_score, bond_status, bond_bg, "#34d399")
    env_points += bond_score; env_max += 12.0

    if data.get("missing_vix", False):
        vix_score = 5.0
        vix_status, vix_bg = "缺失", "bg-yellow"
    else:
        if vix <= 14:
            vix_score, vix_status, vix_bg = 10.0, "风险偏好较好", "bg-green"
        elif vix <= 20:
            vix_score = smooth(10, 7, vix, 14, 20)
            vix_status, vix_bg = "风险情绪中性", "bg-yellow"
        elif vix <= 28:
            vix_score = smooth(7, 3, vix, 20, 28)
            vix_status, vix_bg = "风险偏好降温", "bg-yellow"
        else:
            vix_score = smooth(3, 0, vix, 28, 45)
            vix_status, vix_bg = "风险偏好偏弱", "bg-red"
    vix_score = clamp(vix_score, 0, 10)
    env_scores["vix"] = (vix_score, vix_status, vix_bg, "#34d399")
    env_points += vix_score; env_max += 10.0

    regime = infer_regime(ma20, ma60, band=0.015)
    dist20 = (price - ma20) / ma20 * 100.0 if is_finite(ma20) and ma20 > 0 else 0.0
    dist60 = (price - ma60) / ma60 * 100.0 if is_finite(ma60) and ma60 > 0 else 0.0

    if regime == "bull":
        if price >= ma20 and position20 >= 75:
            trend_score = smooth(10, 6, position20, 75, 100)
            trend_status, trend_bg = "趋势强但偏热", "bg-yellow"
        elif price >= ma60:
            trend_score = smooth(14, 18, max(drawdown20, 0.0), 0, 10)
            trend_status, trend_bg = "趋势回踩", "bg-green"
        else:
            trend_score = smooth(8, 3, abs(dist60), 0, 12)
            trend_status, trend_bg = "跌破关键均线", "bg-red"
    elif regime == "bear":
        if price >= ma20:
            trend_score = smooth(8, 10, position20, 40, 80)
            trend_status, trend_bg = "弱反弹", "bg-yellow"
        else:
            trend_score = smooth(5, 0, abs(dist20), 0, 12)
            trend_status, trend_bg = "下行延续", "bg-red"
    else:
        base = 11.0 - abs(dist20) * 1.1
        if position20 < 40:
            base += 2.0
        trend_score = clamp(base, 0, 18)
        trend_status, trend_bg = "震荡整理", ("bg-green" if position20 < 40 else "bg-yellow")
    trend_score = clamp(trend_score, 0, 18)
    tech_scores["trend"] = (trend_score, trend_status, trend_bg, "#34d399")
    tech_points += trend_score; tech_max += 18.0

    if rsi < 35:
        rsi_score = smooth(12, 10, rsi, 20, 35)
        rsi_status, rsi_bg = "偏冷", "bg-green"
    elif rsi <= 55:
        rsi_score = smooth(10, 8, rsi, 35, 55)
        rsi_status, rsi_bg = "中性", "bg-yellow"
    elif rsi <= 68:
        rsi_score = smooth(8, 5, rsi, 55, 68)
        rsi_status, rsi_bg = "偏热", "bg-yellow"
    else:
        rsi_score = smooth(5, 0, rsi, 68, 88)
        rsi_status, rsi_bg = "过热", "bg-red"
    rsi_score = clamp(rsi_score, 0, 12)
    tech_scores["rsi"] = (rsi_score, rsi_status, rsi_bg, "#34d399")
    tech_points += rsi_score; tech_max += 12.0

    pullback = max(drawdown20, 0.0)
    if pullback <= 4:
        pullback_score = smooth(2, 4, pullback, 0, 4)
        pullback_status, pullback_bg = "贴近高位", "bg-red"
    elif pullback <= 12:
        pullback_score = smooth(4, 10, pullback, 4, 12)
        pullback_status, pullback_bg = "健康回踩", "bg-green"
    elif pullback <= 20:
        pullback_score = smooth(10, 7, pullback, 12, 20)
        pullback_status, pullback_bg = "回撤偏深", "bg-yellow"
    else:
        pullback_score = smooth(7, 3, pullback, 20, 35)
        pullback_status, pullback_bg = "深度回撤", "bg-yellow" if regime != "bear" else "bg-red"
    if regime == "bear" and drawdown60 > 25:
        pullback_score = max(pullback_score - 2.0, 0.0)
    pullback_score = clamp(pullback_score, 0, 10)
    tech_scores["pullback"] = (pullback_score, pullback_status, pullback_bg, "#34d399")
    tech_points += pullback_score; tech_max += 10.0

    if position20 <= 25:
        position_score, position_status, position_bg = 10.0, "区间低位", "bg-green"
    elif position20 <= 50:
        position_score = smooth(10, 7, position20, 25, 50)
        position_status, position_bg = "中低位置", "bg-green"
    elif position20 <= 75:
        position_score = smooth(7, 3, position20, 50, 75)
        position_status, position_bg = "中高位置", "bg-yellow"
    else:
        position_score = smooth(3, 0, position20, 75, 100)
        position_status, position_bg = "接近高位", "bg-red"
    position_score = clamp(position_score, 0, 10)
    tech_scores["position"] = (position_score, position_status, position_bg, "#34d399")
    tech_points += position_score; tech_max += 10.0

    environment_score = normalize_component_score(env_points, env_max)
    technical_score = normalize_component_score(tech_points, tech_max)
    composite_score = clamp(environment_score * 0.4 + technical_score * 0.6, 0, 100)

    environment_label = "风险偏好友好" if environment_score >= 65 else ("环境中性" if environment_score >= 50 else "偏谨慎")
    if regime == "bull" and position20 >= 80 and rsi >= 68:
        position_label = "高位过热区"
    elif regime == "bull" and drawdown20 >= 4 and position20 <= 60:
        position_label = "趋势回踩区"
    elif regime == "bull":
        position_label = "趋势上行区"
    elif regime == "range" and position20 < 35:
        position_label = "震荡低位区"
    elif regime == "range":
        position_label = "高位震荡区"
    else:
        position_label = "弱势回落区"

    if position20 > 90 and rsi >= 78:
        action_label, chase_label = "适合逢高减仓", "短线过热，优先管住追高冲动"
    elif regime == "bull" and position20 >= 80 and rsi >= 70:
        action_label, chase_label = "适合持有", "不建议追高，等待回踩更舒服"
    elif regime == "bull" and drawdown20 >= 4 and drawdown20 <= 12 and rsi <= 60 and environment_score >= 50:
        action_label, chase_label = "适合分批低吸", "更像顺势回踩，不像结构性转弱"
    elif regime == "range" and position20 < 30 and rsi < 45 and environment_score >= 55:
        action_label, chase_label = "适合买入", "可小仓试探，不宜一次性冲满"
    elif regime == "bear" and environment_score < 45:
        action_label, chase_label = "适合观望", "先等风险偏好和均线结构修复"
    elif composite_score >= 62:
        action_label, chase_label = "适合持有", "更适合持有或等回踩，不要追情绪"
    else:
        action_label, chase_label = "适合观望", "高波动资产更适合等更舒服的位置"

    if regime == "bull" and 35 <= position20 <= 75 and 42 <= rsi <= 68 and environment_score >= 50:
        swing_label = "适合"
    elif regime == "bear" or environment_score < 45 or (not data.get("missing_vix", False) and vix >= 30):
        swing_label = "不适合"
    else:
        swing_label = "一般"

    explanations = [
        f"当前环境评分 {environment_score:.1f} / 100，整体属于“{environment_label}”。弱美元、较低利率和更平稳的风险情绪，通常更利于 BTC 维持风险偏好。",
        f"均线结构显示 BTC 更像“{position_label}”。它和纳指不同，波动更大，所以即便方向没坏，也要更关注追高成本。",
        f"20 日分位约 {position20:.0f} / 100，RSI 约 {rsi:.1f}。这会直接影响短线追涨/低吸的舒适度。",
        f"当前动作建议：{action_label}。{chase_label}。",
        f"波段建议：{swing_label}。BTC 更适合看节奏和仓位控制，而不是把单次信号当成绝对命令。",
    ]

    return {
        "environment_scores": env_scores,
        "technical_scores": tech_scores,
        "environment_score": environment_score,
        "technical_score": technical_score,
        "composite_score": composite_score,
        "environment_label": environment_label,
        "position_label": position_label,
        "action_label": action_label,
        "swing_label": swing_label,
        "chase_label": chase_label,
        "regime": regime,
        "explanations": explanations,
    }




def calculate_hstech_view(data: dict) -> dict:
    env_scores = {}
    tech_scores = {}

    price = safe_float(data.get("price"))
    ma20 = safe_float(data.get("ma20"))
    ma60 = safe_float(data.get("ma60"))
    rsi = safe_float(data.get("rsi"))
    drawdown20 = safe_float(data.get("drawdown20"))
    drawdown60 = safe_float(data.get("drawdown60"))
    position20 = safe_float(data.get("position20"))
    us10y = safe_float(data.get("us10y"))
    dxy = safe_float(data.get("dxy"))
    usd_cny = safe_float(data.get("usd_cny"))

    env_points = env_max = tech_points = tech_max = 0.0

    if data.get("missing_dxy", False):
        dxy_score = 10.0
        dxy_status, dxy_bg = "缺失", "bg-yellow"
    else:
        if dxy <= 100:
            dxy_score, dxy_status, dxy_bg = 20.0, "弱美元", "bg-green"
        elif dxy <= 104:
            dxy_score = smooth(20, 14, dxy, 100, 104)
            dxy_status, dxy_bg = "美元中性", "bg-yellow"
        elif dxy <= 107:
            dxy_score = smooth(14, 7, dxy, 104, 107)
            dxy_status, dxy_bg = "美元偏强", "bg-yellow"
        else:
            dxy_score = smooth(7, 0, dxy, 107, 112)
            dxy_status, dxy_bg = "美元压制", "bg-red"
    dxy_score = clamp(dxy_score, 0, 20)
    env_scores["dxy"] = (dxy_score, dxy_status, dxy_bg, "#34d399")
    env_points += dxy_score
    env_max += 20.0

    if data.get("missing_us10y", False):
        bond_score = 7.0
        bond_status, bond_bg = "缺失", "bg-yellow"
    else:
        if us10y <= 3.8:
            bond_score, bond_status, bond_bg = 15.0, "利率友好", "bg-green"
        elif us10y <= 4.3:
            bond_score = smooth(15, 10, us10y, 3.8, 4.3)
            bond_status, bond_bg = "利率中性", "bg-yellow"
        elif us10y <= 4.9:
            bond_score = smooth(10, 5, us10y, 4.3, 4.9)
            bond_status, bond_bg = "利率偏高", "bg-yellow"
        else:
            bond_score = smooth(5, 0, us10y, 4.9, 6.0)
            bond_status, bond_bg = "高利率压制", "bg-red"
    bond_score = clamp(bond_score, 0, 15)
    env_scores["bond"] = (bond_score, bond_status, bond_bg, "#34d399")
    env_points += bond_score
    env_max += 15.0

    if data.get("missing_usd_cny", False):
        cny_score = 7.0
        cny_status, cny_bg = "缺失", "bg-yellow"
    else:
        if usd_cny <= 7.10:
            cny_score, cny_status, cny_bg = 15.0, "人民币偏稳", "bg-green"
        elif usd_cny <= 7.22:
            cny_score = smooth(15, 11, usd_cny, 7.10, 7.22)
            cny_status, cny_bg = "汇率中性", "bg-yellow"
        elif usd_cny <= 7.32:
            cny_score = smooth(11, 6, usd_cny, 7.22, 7.32)
            cny_status, cny_bg = "人民币承压", "bg-yellow"
        else:
            cny_score = smooth(6, 0, usd_cny, 7.32, 7.55)
            cny_status, cny_bg = "汇率压力较大", "bg-red"
    cny_score = clamp(cny_score, 0, 15)
    env_scores["cny"] = (cny_score, cny_status, cny_bg, "#34d399")
    env_points += cny_score
    env_max += 15.0

    regime = infer_regime(ma20, ma60, band=0.012)
    dist20 = (price - ma20) / ma20 * 100.0 if is_finite(ma20) and ma20 > 0 else 0.0
    dist60 = (price - ma60) / ma60 * 100.0 if is_finite(ma60) and ma60 > 0 else 0.0

    if regime == "bull":
        if position20 >= 80:
            trend_score = smooth(12, 8, position20, 80, 100)
            trend_status, trend_bg = "趋势增强但偏热", "bg-yellow"
        elif price >= ma60:
            trend_score = smooth(14, 20, max(drawdown20, 0.0), 0, 12)
            trend_status, trend_bg = "低位修复", "bg-green"
        else:
            trend_score = smooth(10, 4, abs(dist60), 0, 10)
            trend_status, trend_bg = "失守中期均线", "bg-red"
    elif regime == "bear":
        if price >= ma20:
            trend_score = smooth(8, 10, position20, 30, 70)
            trend_status, trend_bg = "弱修复", "bg-yellow"
        else:
            trend_score = smooth(5, 0, abs(dist20), 0, 12)
            trend_status, trend_bg = "弱势震荡", "bg-red"
    else:
        base = 12.0 - abs(dist20) * 1.5
        if position20 < 35:
            base += 2.0
        trend_score = clamp(base, 0, 20)
        trend_status, trend_bg = "震荡观察", ("bg-green" if position20 < 35 else "bg-yellow")
    trend_score = clamp(trend_score, 0, 20)
    tech_scores["trend"] = (trend_score, trend_status, trend_bg, "#34d399")
    tech_points += trend_score
    tech_max += 20.0

    if rsi < 35:
        rsi_score = smooth(12, 10, rsi, 22, 35)
        rsi_status, rsi_bg = "偏冷", "bg-green"
    elif rsi <= 55:
        rsi_score = smooth(10, 8, rsi, 35, 55)
        rsi_status, rsi_bg = "中性", "bg-yellow"
    elif rsi <= 68:
        rsi_score = smooth(8, 5, rsi, 55, 68)
        rsi_status, rsi_bg = "偏热", "bg-yellow"
    else:
        rsi_score = smooth(5, 0, rsi, 68, 86)
        rsi_status, rsi_bg = "过热", "bg-red"
    rsi_score = clamp(rsi_score, 0, 12)
    tech_scores["rsi"] = (rsi_score, rsi_status, rsi_bg, "#34d399")
    tech_points += rsi_score
    tech_max += 12.0

    pullback = max(drawdown20, 0.0)
    if pullback <= 5:
        pullback_score = smooth(1, 4, pullback, 0, 5)
        pullback_status, pullback_bg = "接近区间高位", "bg-red"
    elif pullback <= 14:
        pullback_score = smooth(4, 10, pullback, 5, 14)
        pullback_status, pullback_bg = "回调观察区", "bg-green"
    elif pullback <= 24:
        pullback_score = smooth(10, 7, pullback, 14, 24)
        pullback_status, pullback_bg = "回撤偏深", "bg-yellow"
    else:
        pullback_score = smooth(7, 3, pullback, 24, 40)
        pullback_status, pullback_bg = "深度回撤", "bg-yellow" if regime != "bear" else "bg-red"
    if regime == "bear" and drawdown60 > 20:
        pullback_score = max(pullback_score - 2.0, 0.0)
    pullback_score = clamp(pullback_score, 0, 10)
    tech_scores["pullback"] = (pullback_score, pullback_status, pullback_bg, "#34d399")
    tech_points += pullback_score
    tech_max += 10.0

    if position20 <= 25:
        position_score = 8.0 if regime != "bear" else 4.0
        position_status, position_bg = "区间低位", "bg-green" if regime != "bear" else "bg-yellow"
    elif position20 <= 50:
        position_score = smooth(8, 6, position20, 25, 50)
        position_status, position_bg = "中低位置", "bg-green"
    elif position20 <= 75:
        position_score = smooth(6, 2, position20, 50, 75)
        position_status, position_bg = "中高位置", "bg-yellow"
    else:
        position_score = smooth(2, 0, position20, 75, 100)
        position_status, position_bg = "接近高位", "bg-red"
    position_score = clamp(position_score, 0, 8)
    tech_scores["position"] = (position_score, position_status, position_bg, "#34d399")
    tech_points += position_score
    tech_max += 8.0

    environment_score = normalize_component_score(env_points, env_max)
    technical_score = normalize_component_score(tech_points, tech_max)
    composite_score = clamp(environment_score * 0.45 + technical_score * 0.55, 0, 100)

    environment_label = "流动性偏友好" if environment_score >= 65 else ("中性偏等待" if environment_score >= 50 else "偏谨慎")
    if regime == "bull" and position20 >= 75 and rsi >= 68:
        position_label = "趋势增强但不宜追高"
    elif regime == "bull" and position20 <= 60:
        position_label = "低位修复区"
    elif regime == "range" and position20 < 35:
        position_label = "左侧观察区"
    elif regime == "bear":
        position_label = "弱势震荡区"
    else:
        position_label = "反弹观察区"

    if regime == "bear" and environment_score < 45 and position20 < 35:
        action_label, left_side_label = "不适合抄底", "趋势未扭转，左侧只适合继续观察"
    elif regime == "bull" and position20 > 80 and rsi >= 70:
        action_label, left_side_label = "不建议追高", "更适合等回踩，不适合情绪化追涨"
    elif regime == "bull" and position20 <= 65 and rsi < 65 and environment_score >= 50:
        action_label, left_side_label = "适合分批布局", "可以按仓位分层介入，而不是一次性打满"
    elif composite_score >= 65:
        action_label, left_side_label = "适合持有", "更像修复延续，持有优先于频繁换手"
    elif composite_score >= 55 and position20 < 45:
        action_label, left_side_label = "适合买入", "更偏左侧试探，不宜把它当成确定性趋势单"
    elif environment_score < 45:
        action_label, left_side_label = "适合控制仓位", "宏观环境不友好，仓位节奏比抄底冲动更重要"
    else:
        action_label, left_side_label = "适合观望", "先等环境或趋势给出更清晰的改善信号"

    explanations = [
        f"当前环境评分 {environment_score:.1f} / 100，整体属于“{environment_label}”。恒生科技对美元、利率和汇率变化通常更敏感。",
        f"当前价格使用 ETF 代理，不等同于恒生科技指数点位本体；更适合看方向、位置和节奏，而不是拿它当精确指数报价。",
        f"均线与分位显示当前更像“{position_label}”。跌得多不等于马上适合抄底，关键还要看趋势有没有真正修复。",
        f"当前动作建议：{action_label}。{left_side_label}。",
        f"20 日分位约 {position20:.0f} / 100，RSI 约 {rsi:.1f}，更适合把它当成高弹性风险资产而不是港股版纳指。",
    ]

    return {
        "environment_scores": env_scores,
        "technical_scores": tech_scores,
        "environment_score": environment_score,
        "technical_score": technical_score,
        "composite_score": composite_score,
        "environment_label": environment_label,
        "position_label": position_label,
        "action_label": action_label,
        "left_side_label": left_side_label,
        "regime": regime,
        "explanations": explanations,
    }


def build_btc_status_lines(data: dict, quality_label: str, quality_reason: str) -> list[str]:
    fallback_text = "是" if data.get("used_fallback_data_source", False) else "否"
    return [
        f"主要数据源：{source_label(data.get('data_source'))}",
        f"是否使用 fallback：{fallback_text}",
        format_metric_status(
            "BTC 人民币参考价",
            data.get("btc_cny_price"),
            data.get("missing_btc_cny_price", False),
            data.get("btc_cny_source_label"),
            digits=0,
            suffix=" 元/枚",
            missing_note="缺失（缺少 USD/CNY）",
        ),
        format_metric_status(
            "BTC 美元参考价",
            data.get("price"),
            False,
            data.get("price_source_label"),
            digits=0,
            suffix=" 美元/枚",
            missing_note="缺失",
        ),
        format_metric_status(
            "USD/CNY",
            data.get("usd_cny"),
            data.get("missing_usd_cny", False),
            data.get("usd_cny_source_label"),
            digits=4,
            missing_note="缺失",
        ),
        f"换算公式：{data.get('btc_cny_formula_label')}",
        f"换算可信度：{data.get('btc_cny_quality_label')}（{data.get('btc_cny_quality_reason')}）",
        format_metric_status(
            "VIX",
            data.get("vix"),
            data.get("missing_vix", False),
            data.get("vix_source_label"),
            missing_note="缺失（未使用代理值）",
        ),
        format_metric_status(
            "DXY",
            data.get("dxy"),
            data.get("missing_dxy", False),
            data.get("dxy_source_label"),
            missing_note="缺失（未使用代理值）",
        ),
        format_metric_status(
            "10Y 美债收益率",
            data.get("us10y"),
            data.get("missing_us10y", False),
            data.get("us10y_source_label"),
            suffix="%",
            missing_note="缺失",
        ),
        f"当前结果可信度：{quality_label}（{quality_reason}）",
    ]

def build_hstech_status_lines(data: dict, quality_label: str, quality_reason: str) -> list[str]:
    fallback_text = "是" if data.get("used_fallback_data_source", False) else "否"
    hsi_line = (
        "恒指方向辅助项：缺失"
        if data.get("missing_hsi", False)
        else f"恒指方向辅助项：{data.get('hsi_regime_text')}（{data.get('hsi_source_label')}）"
    )
    return [
        f"主要数据源：{source_label(data.get('data_source'))}",
        f"是否使用 fallback：{fallback_text}",
        "主价格说明：当前展示的是恒生科技 ETF 代理价，不等同于恒生科技指数点位本体",
        format_metric_status(
            "恒生科技价格代理",
            data.get("price"),
            False,
            data.get("price_source_label"),
            digits=2,
            suffix=" HK$",
            is_proxy=True,
            missing_note="缺失",
        ),
        format_metric_status(
            "USD/CNY",
            data.get("usd_cny"),
            data.get("missing_usd_cny", False),
            data.get("usd_cny_source_label"),
            digits=4,
            missing_note="缺失",
        ),
        format_metric_status(
            "DXY",
            data.get("dxy"),
            data.get("missing_dxy", False),
            data.get("dxy_source_label"),
            missing_note="缺失（未使用代理值）",
        ),
        format_metric_status(
            "10Y 美债收益率",
            data.get("us10y"),
            data.get("missing_us10y", False),
            data.get("us10y_source_label"),
            suffix="%",
            missing_note="缺失",
        ),
        hsi_line,
        f"当前结果可信度：{quality_label}（{quality_reason}）",
    ]


def render_btc_panel() -> None:
    with st.spinner("正在获取比特币数据..."):
        data = get_btc_market_data()
    if data is None:
        st.error("无法获取比特币数据：请检查网络连接，或确认能访问 Yahoo Finance / Stooq。")
        return

    view = calculate_btc_view(data)
    quality_label, quality_reason = build_data_quality_summary(data)
    status_lines = build_btc_status_lines(data, quality_label, quality_reason)
    rec_class, icon = action_rec_style(view["action_label"], "₿")
    composite_score = safe_float(view.get("composite_score"))
    usd_price_text = f"${format_number(data.get('price'), digits=0)} / 枚"
    cny_price_text = "—" if data.get("missing_btc_cny_price", False) else f"¥{format_number(data.get('btc_cny_price'), digits=0)} / 枚"
    position_text = format_number(data.get("position20"), digits=0, suffix=" / 100")
    usd_cny_text = "—" if data.get("missing_usd_cny", False) else format_number(data.get("usd_cny"), digits=4)
    st.markdown("<h3 class='section-title'>₿ 比特币决策面板</h3>", unsafe_allow_html=True)
    price_mode = st.radio("参考价显示", ["人民币 / 枚", "美元 / 枚"], horizontal=True, key="btc_price_mode")
    selected_price_text = cny_price_text if price_mode == "人民币 / 枚" else usd_price_text
    secondary_price_text = usd_price_text if price_mode == "人民币 / 枚" else cny_price_text
    brief_line = f"当前更像{view['position_label']}；{view['chase_label']}。"
    summary_lines = [
        f"波段建议：{view['swing_label']}",
        f"追高提醒：{view['chase_label']}",
        f"20 日分位：{position_text}",
        f"USD/CNY：{usd_cny_text}",
        f"另一口径参考价：{secondary_price_text}",
    ]

    render_summary_chart_section(
        data,
        f"BTC 走势（当前：${format_number(data.get('price'), digits=0)}）",
        "BTC 综合分",
        composite_score,
        view["action_label"],
        view["environment_label"],
        f"参考价：{selected_price_text}",
        brief_line,
        summary_lines,
        rec_class,
        icon,
    )

    st.markdown("<h3 class='section-title'>🧭 比特币为什么这样判断</h3>", unsafe_allow_html=True)
    render_explanation_box("判断逻辑", view["explanations"])

    st.markdown("<h3 class='section-title'>📊 比特币核心因子</h3>", unsafe_allow_html=True)
    row1 = st.columns(4)
    env_bg = "bg-green" if view["environment_score"] >= 60 else ("bg-yellow" if view["environment_score"] >= 45 else "bg-red")
    tech_bg = "bg-green" if view["technical_score"] >= 60 else ("bg-yellow" if view["technical_score"] >= 45 else "bg-red")
    swing_bg = "bg-green" if view["swing_label"] == "适合" else ("bg-yellow" if view["swing_label"] == "一般" else "bg-red")
    with row1[0]:
        render_info_card("当前参考价", selected_price_text, f"切换口径：{price_mode}；另一口径 {secondary_price_text}", "参考价", "bg-yellow")
    with row1[1]:
        render_info_card("环境分", format_number(view["environment_score"], digits=1), "越高表示风险偏好和流动性背景越友好", view["environment_label"], env_bg)
    with row1[2]:
        render_info_card("技术分", format_number(view["technical_score"], digits=1), "越高表示当前位置和节奏更舒服", view["position_label"], tech_bg)
    with row1[3]:
        render_info_card("波段建议", view["swing_label"], "只辅助判断节奏，不代替仓位纪律", view["swing_label"], swing_bg)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    row2 = st.columns(4)
    with row2[0]:
        render_card("美元指数 DXY", "—" if data.get("missing_dxy", False) else format_number(data.get("dxy")), "弱美元通常更利于 BTC 风险偏好" if not data.get("missing_dxy", False) else "缺失（未使用代理）", view["environment_scores"]["dxy"], 18)
    with row2[1]:
        render_card("10年美债收益率", "—" if data.get("missing_us10y", False) else format_number(data.get("us10y"), suffix="%"), "利率越低，流动性背景通常越友好" if not data.get("missing_us10y", False) else "缺失", view["environment_scores"]["bond"], 12)
    with row2[2]:
        render_card("VIX / 风险情绪", "—" if data.get("missing_vix", False) else format_number(data.get("vix")), "VIX 越低，风险偏好通常越稳定" if not data.get("missing_vix", False) else "缺失（未使用代理）", view["environment_scores"]["vix"], 10)
    with row2[3]:
        render_info_card("USD/CNY", usd_cny_text, "人民币参考价换算所用汇率", data.get("btc_cny_quality_label"), "bg-green" if data.get("btc_cny_quality_label") == "高" else ("bg-yellow" if data.get("btc_cny_quality_label") == "中" else "bg-red"))

    st.markdown("<h3 class='section-title'>🔬 比特币技术面与数据状态</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    row3 = st.columns(4)
    with row3[0]:
        render_card("RSI (14)", format_number(data.get("rsi"), digits=1), "越高越容易出现追高风险", view["technical_scores"]["rsi"], 12)
    with row3[1]:
        render_card("趋势得分", format_number(view["technical_scores"]["trend"][0]), "MA20 / MA60 结构", view["technical_scores"]["trend"], 18)
    with row3[2]:
        render_card("回撤位置", f"20日 -{format_number(data.get('drawdown20'))}% ｜ 60日 -{format_number(data.get('drawdown60'))}%", "回踩更舒服，但要配合趋势看", view["technical_scores"]["pullback"], 10)
    with row3[3]:
        render_card("20日价格分位", position_text, "越靠上方越不适合追高", view["technical_scores"]["position"], 10)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    render_data_status_section("📎 比特币数据状态", status_lines, quality_label, quality_reason)

def render_hstech_panel() -> None:
    with st.spinner("正在获取恒生科技数据..."):
        data = get_hstech_market_data()
    if data is None:
        st.error("无法获取恒生科技数据：请检查网络连接，或确认能访问 Yahoo Finance / Stooq。")
        return

    view = calculate_hstech_view(data)
    quality_label, quality_reason = build_data_quality_summary(data)
    status_lines = build_hstech_status_lines(data, quality_label, quality_reason)
    rec_class, icon = action_rec_style(view["action_label"], "🇭🇰")
    composite_score = safe_float(view.get("composite_score"))
    price_text = format_number(data.get("price"), digits=2, suffix=" HK$")
    position_text = format_number(data.get("position20"), digits=0, suffix=" / 100")
    usd_cny_text = "—" if data.get("missing_usd_cny", False) else format_number(data.get("usd_cny"), digits=4)
    hsi_direction_text = "缺失" if data.get("missing_hsi", False) else data.get("hsi_regime_text")
    brief_line = f"当前更像{view['position_label']}；{view['left_side_label']}。"
    summary_lines = [
        f"代理价：{price_text}",
        f"动作建议：{view['action_label']}",
        f"左侧提示：{view['left_side_label']}",
        f"USD/CNY：{usd_cny_text}",
        f"恒指方向辅助项：{hsi_direction_text}",
        f"20 日分位：{position_text}",
    ]

    st.markdown("<h3 class='section-title'>🇭🇰 恒生科技决策面板</h3>", unsafe_allow_html=True)
    render_summary_chart_section(
        data,
        f"恒生科技代理走势（当前：{price_text}）",
        "恒生科技综合分",
        composite_score,
        view["action_label"],
        view["environment_label"],
        f"代理价：{price_text}",
        brief_line,
        summary_lines,
        rec_class,
        icon,
    )

    st.markdown("<h3 class='section-title'>🧭 恒生科技为什么这样判断</h3>", unsafe_allow_html=True)
    render_explanation_box("判断逻辑", view["explanations"])

    st.markdown("<h3 class='section-title'>📊 恒生科技核心因子</h3>", unsafe_allow_html=True)
    row1 = st.columns(4)
    env_bg = "bg-green" if view["environment_score"] >= 60 else ("bg-yellow" if view["environment_score"] >= 45 else "bg-red")
    tech_bg = "bg-green" if view["technical_score"] >= 60 else ("bg-yellow" if view["technical_score"] >= 45 else "bg-red")
    action_bg = "bg-green" if view["action_label"] in {"适合买入", "适合分批布局", "适合持有"} else ("bg-yellow" if view["action_label"] in {"适合观望", "不建议追高"} else "bg-red")
    hsi_bg = "bg-green" if hsi_direction_text == "上升趋势" else ("bg-yellow" if hsi_direction_text == "震荡区间" else "bg-red")
    with row1[0]:
        render_info_card("价格代理", price_text, "ETF 代理价，不等同于恒生科技指数点位", "代理", "bg-yellow")
    with row1[1]:
        render_info_card("环境分", format_number(view["environment_score"], digits=1), "越高表示美元/利率/汇率背景越友好", view["environment_label"], env_bg)
    with row1[2]:
        render_info_card("技术分", format_number(view["technical_score"], digits=1), "越高表示修复结构更清晰", view["position_label"], tech_bg)
    with row1[3]:
        render_info_card("动作建议", view["action_label"], "高弹性资产更适合分批布局，不适合情绪化抄底", view["action_label"], action_bg)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    row2 = st.columns(4)
    with row2[0]:
        render_card("美元指数 DXY", "—" if data.get("missing_dxy", False) else format_number(data.get("dxy")), "美元越弱，通常越利于港股科技情绪" if not data.get("missing_dxy", False) else "缺失（未使用代理）", view["environment_scores"]["dxy"], 20)
    with row2[1]:
        render_card("10年美债收益率", "—" if data.get("missing_us10y", False) else format_number(data.get("us10y"), suffix="%"), "高利率环境通常不利于高弹性成长资产" if not data.get("missing_us10y", False) else "缺失", view["environment_scores"]["bond"], 15)
    with row2[2]:
        render_card("USD/CNY", usd_cny_text, "人民币更稳通常更利于风险偏好" if not data.get("missing_usd_cny", False) else "缺失", view["environment_scores"]["cny"], 15)
    with row2[3]:
        render_info_card("恒指方向辅助项", hsi_direction_text, "用于补充观察港股大盘方向，不直接参与主评分", hsi_direction_text, hsi_bg if not data.get("missing_hsi", False) else "bg-red")

    st.markdown("<h3 class='section-title'>🔬 恒生科技技术面与数据状态</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    row3 = st.columns(4)
    with row3[0]:
        render_card("RSI (14)", format_number(data.get("rsi"), digits=1), "越高越容易进入追高区", view["technical_scores"]["rsi"], 12)
    with row3[1]:
        render_card("趋势得分", format_number(view["technical_scores"]["trend"][0]), "MA20 / MA60 结构", view["technical_scores"]["trend"], 20)
    with row3[2]:
        render_card("回撤位置", f"20日 -{format_number(data.get('drawdown20'))}% ｜ 60日 -{format_number(data.get('drawdown60'))}%", "跌得多不等于就能直接抄底", view["technical_scores"]["pullback"], 10)
    with row3[3]:
        render_card("20日价格分位", position_text, "越低越适合左侧观察，但要配合趋势", view["technical_scores"]["position"], 8)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    render_data_status_section("📎 恒生科技数据状态", status_lines, quality_label, quality_reason)


# ==========================================
# 7. Main page
# ==========================================

# ==========================================
# 7. Main page
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<h1 style="color:#34d399; margin-bottom:0;">📊 多资产投资决策台</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#64748b; margin-top:5px;">wyh构建的查看纳斯达克 100、黄金、比特币与恒生科技的环境、位置、动作建议和数据可信度数据看板。仅作辅助决策，不构成投资建议。</p>',
        unsafe_allow_html=True,
    )

with col_h2:
    if st.button("🔄 刷新全部数据"):
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

st.markdown("---")

tab_nasdaq, tab_gold, tab_btc, tab_hstech = st.tabs(["📈 纳斯达克 100", "🥇 黄金", "₿ 比特币", "🇭🇰 恒生科技"])
with tab_nasdaq:
    render_nasdaq_panel()
with tab_gold:
    render_gold_panel()
with tab_btc:
    render_btc_panel()
with tab_hstech:
    render_hstech_panel()
