import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone


# 统一使用北京时间（UTC+8）
def now_cn():
    """返回北京时间（UTC+8）"""
    return datetime.now(timezone.utc) + timedelta(hours=8)


# ==========================================
# 1. 页面配置与 CSS 样式注入 (打造漂亮 UI)
# ==========================================
st.set_page_config(page_title="纳斯达克量化决策台", page_icon="📈", layout="wide")

# 自定义 CSS，复刻之前 HTML 版的深色现代风格
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* 隐藏 Streamlit 默认头部 */
    header {visibility: hidden;}
    
    /* 卡片容器样式 */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        transition: transform 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        border-color: #475569;
    }
    
    /* 文字样式 */
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
    
    /* 状态标签 */
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
    
    /* 进度条背景 */
    .progress-bg {
        background-color: #334155;
        height: 6px;
        border-radius: 3px;
        margin-top: 10px;
        overflow: hidden;
    }
    
    /* 建议卡片 */
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
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心数据获取 (使用 yfinance)
# ==========================================
@st.cache_data(ttl=600)  # 缓存10分钟，避免刷新太频繁被封IP
def get_market_data():
    try:
        # 1. 纳指价格与技术指标 (^NDX)
        ndx = yf.Ticker("^NDX")
        hist = ndx.history(period="1y")
        
        current_price = hist['Close'].iloc[-1]
        
        # 均线
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(window=60).mean().iloc[-1]
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 回撤
        rolling_max = hist['Close'].rolling(window=252, min_periods=1).max()
        drawdown = abs((hist['Close'] / rolling_max - 1.0).iloc[-1] * 100)

        # 2. 市盈率 (使用 QQQ 作为 NDX 的替代，因为指数 PE 很难直接获取)
        qqq = yf.Ticker("QQQ")
        pe = qqq.info.get('trailingPE', 30)  # 如果获取失败，默认为30防止报错
        if pe is None:
            pe = 30

        # 3. 其他宏观指标
        vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
        us10y = yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]  # 单位是收益率，例如 4.2
        dxy = yf.Ticker("DX-Y.NYB").history(period="5d")['Close'].iloc[-1]

        return {
            "price": current_price, "ma20": ma20, "ma60": ma60,
            "rsi": rsi, "drawdown": drawdown, "pe": pe,
            "vix": vix, "us10y": us10y, "dxy": dxy,
            "history": hist
        }
    except Exception:
        return None

# ==========================================
# 3. 打分逻辑函数（混合型：估值 + 趋势 + 回撤 + 情绪 + 宏观）
# ==========================================
def smooth(low_score, high_score, x, x_low, x_high):
    """
    连续化线性插值：
    - x <= x_low  -> low_score
    - x >= x_high -> high_score
    - 中间线性插值
    """
    if x <= x_low:
        return float(low_score)
    if x >= x_high:
        return float(high_score)
    ratio = (x - x_low) / (x_high - x_low)
    return float(low_score + ratio * (high_score - low_score))


def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


def calculate_score(data):
    """
    连续化小数打分（总分理论上<=100，且做硬性 cap）
    各因子最大分与 UI 进度条一致：
      PE 25, Trend 20, DD 20, RSI 7, VIX 8, Bond 10, DXY 10  -> 合计 100
    """
    scores = {}

    # 数据
    p = float(data['price'])
    ma20 = float(data['ma20'])
    ma60 = float(data['ma60'])
    rsi = float(data['rsi'])
    d = float(data['drawdown'])
    v = float(data['vix'])
    u = float(data['us10y'])
    dx = float(data['dxy'])
    pe = float(data['pe'])

    # ========== 1) PE（保留原档位含义，但档内连续化；范围 [5,25]） ==========
    if pe < 22:
        pe_score = smooth(22, 25, pe, 15, 22)
        scores['pe'] = (pe_score, '极低估', 'bg-green', '#34d399')
    elif pe < 25:
        pe_score = smooth(20, 22, pe, 22, 25)
        scores['pe'] = (pe_score, '低估', 'bg-green', '#34d399')
    elif pe < 28:
        pe_score = smooth(15, 20, pe, 25, 28)
        scores['pe'] = (pe_score, '合理', 'bg-yellow', '#fbbf24')
    elif pe < 32:
        pe_score = smooth(10, 15, pe, 28, 32)
        scores['pe'] = (pe_score, '偏高', 'bg-yellow', '#fbbf24')
    else:
        pe_score = smooth(5, 10, pe, 32, 45)
        scores['pe'] = (pe_score, '高估泡沫', 'bg-red', '#fb7185')

    # ========== regime：牛/熊/震荡 ==========
    if ma20 > ma60 * 1.01:
        regime = 'bull'
    elif ma20 < ma60 * 0.99:
        regime = 'bear'
    else:
        regime = 'range'

    # 价格相对均线的偏离（%）
    dist20 = (p - ma20) / ma20 * 100.0
    dist60 = (p - ma60) / ma60 * 100.0

    # ========== 2) 趋势 Trend（范围 [0,20]） ==========
    if regime == 'bull':
        if p >= ma20:
            # 越高于 MA20 越偏“追涨风险”，分数从 18 -> 12
            trend_score = smooth(18, 12, dist20, 0, 5)
            status = '强势上行'
            bg = 'bg-yellow'
        elif p >= ma60:
            # 回调区：越接近 MA60（更深回调）越接近 20
            trend_score = smooth(16, 20, -dist20, 0, 5)
            status = '上升回调（偏买点）'
            bg = 'bg-green'
        else:
            # 跌破 MA60：越深越低
            trend_score = smooth(12, 4, -dist60, 0, 10)
            status = '跌破关键均线'
            bg = 'bg-red'
    elif regime == 'bear':
        if p >= ma20:
            # 空头反弹：越高越像“减仓窗口”，分数不高
            trend_score = smooth(8, 10, dist20, 0, 5)
            status = '空头反弹'
            bg = 'bg-yellow'
        elif p >= ma60:
            trend_score = smooth(4, 8, dist60, 0, 5)
            status = '弱反弹'
            bg = 'bg-yellow'
        else:
            trend_score = smooth(4, 0, -dist20, 0, 10)
            status = '下跌延续'
            bg = 'bg-red'
    else:
        # 震荡：靠近 MA20 中轴更安全；在下沿（dist20<0）略加分
        base = 14.0 - abs(dist20) * 2.0
        if dist20 < 0:
            base += 2.0
        else:
            base -= 1.0
        trend_score = clamp(base, 0, 20)
        status = '震荡区间'
        bg = 'bg-yellow' if abs(dist20) < 2 else ('bg-green' if dist20 < 0 else 'bg-yellow')

    scores['trend'] = (clamp(trend_score, 0, 20), status, bg, '#34d399')

        # ========== 3) 回撤 DD（范围 [0,20]；连续化但对齐旧分档锚点） ==========
    # 目标：连续化后仍接近旧版分布（避免常态下被系统性压低）
    # 旧版锚点大致：0~5%≈5分；5~10%≈10分；10~15%≈15分；>=15%≈20分
    if d <= 0:
        dd_score = 0.0
    elif d < 5:
        dd_score = smooth(5.0, 10.0, d, 0.0, 5.0)
    elif d < 10:
        dd_score = smooth(10.0, 15.0, d, 5.0, 10.0)
    elif d < 15:
        dd_score = smooth(15.0, 20.0, d, 10.0, 15.0)
    else:
        dd_score = 20.0

    if d >= 15:
        dd_status, dd_bg = '深度回撤', 'bg-green'
    elif d >= 10:
        dd_status, dd_bg = '中度回撤', 'bg-green'
    elif d >= 5:
        dd_status, dd_bg = '轻微回撤', 'bg-yellow'
    elif d > 0:
        dd_status, dd_bg = '微跌', 'bg-yellow'
    else:
        dd_status, dd_bg = '新高附近', 'bg-red'

    scores['dd'] = (clamp(dd_score, 0, 20), dd_status, dd_bg, '#34d399')

    # ========== 4) RSI（范围 [0,7]） ==========
    if rsi < 30:
        rsi_score = smooth(7, 5.5, rsi, 10, 30)
        rsi_status, rsi_bg = '超卖', 'bg-green'
    elif rsi <= 50:
        rsi_score = smooth(5.5, 4.5, rsi, 30, 50)
        rsi_status, rsi_bg = '偏弱', 'bg-green'
    elif rsi <= 70:
        rsi_score = smooth(4.5, 2.5, rsi, 50, 70)
        rsi_status, rsi_bg = '偏强', 'bg-yellow'
    else:
        rsi_score = smooth(2.5, 0.0, rsi, 70, 90)
        rsi_status, rsi_bg = '超买', 'bg-red'
    scores['rsi'] = (clamp(rsi_score, 0, 7), rsi_status, rsi_bg, '#34d399')

    # ========== 5) VIX（范围 [0,8]；高波动降分） ==========
    if v < 12:
        vix_score = smooth(7, 8, 12 - v, 0, 6)
        vix_status, vix_bg = '低波动', 'bg-green'
    elif v < 20:
        vix_score = smooth(8, 5, v, 12, 20)
        vix_status, vix_bg = '正常波动', 'bg-green'
    elif v < 28:
        vix_score = smooth(5, 2, v, 20, 28)
        vix_status, vix_bg = '波动加大', 'bg-yellow'
    else:
        vix_score = smooth(2, 0, v, 28, 45)
        vix_status, vix_bg = '恐慌区', 'bg-red'
    scores['vix'] = (clamp(vix_score, 0, 8), vix_status, vix_bg, '#34d399')

    # ========== 6) 10Y 美债（范围 [0,10]；利率越高越压分） ==========
    if u <= 3.5:
        bond_score, bond_status, bond_bg = 10.0, '利率友好', 'bg-green'
    elif u <= 4.2:
        bond_score = smooth(10, 7.5, u, 3.5, 4.2)
        bond_status, bond_bg = '中性利率', 'bg-yellow'
    elif u <= 4.8:
        bond_score = smooth(7.5, 4.5, u, 4.2, 4.8)
        bond_status, bond_bg = '偏高利率', 'bg-yellow'
    else:
        bond_score = smooth(4.5, 0.0, u, 4.8, 6.0)
        bond_status, bond_bg = '高利率压制', 'bg-red'
    scores['bond'] = (clamp(bond_score, 0, 10), bond_status, bond_bg, '#34d399')

    # ========== 7) DXY（范围 [0,10]；美元越强越压分） ==========
    if dx <= 100:
        dxy_score, dxy_status, dxy_bg = 10.0, '弱美元利好', 'bg-green'
    elif dx <= 104:
        dxy_score = smooth(10, 7.5, dx, 100, 104)
        dxy_status, dxy_bg = '中性美元', 'bg-yellow'
    elif dx <= 106:
        dxy_score = smooth(7.5, 4.0, dx, 104, 106)
        dxy_status, dxy_bg = '强美元', 'bg-yellow'
    else:
        dxy_score = smooth(4.0, 0.0, dx, 106, 112)
        dxy_status, dxy_bg = '极强美元', 'bg-red'
    scores['dxy'] = (clamp(dxy_score, 0, 10), dxy_status, dxy_bg, '#34d399')

    # ========== 汇总总分（硬 cap：<=100） ==========
    total = float(sum(item[0] for item in scores.values()))
    total = clamp(total, 0, 100)

    return scores, total


# ==========================================
# 4. 辅助 UI 组件渲染函数
# ==========================================
def render_card(title, value, subtext, score_info, max_score):
    score, status, bg_class, bar_color = score_info
    pct = (score / max_score) * 100
    
    st.markdown(f"""
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
            <span>贡献得分</span>
            <span style="font-family:monospace; color:#94a3b8;">{score:.2f}/{max_score}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. 主程序逻辑
# ==========================================

# 顶部标题
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<h1 style="color:#34d399; margin-bottom:0;">🦅 纳斯达克 100 决策台</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; margin-top:5px;">基于 Python 实时数据流的量化分析系统</p>', unsafe_allow_html=True)
with col_h2:
    if st.button("🔄 刷新实时数据"):
        st.cache_data.clear()
        st.rerun()

    # 在刷新按钮下面显示当前数据更新时间（北京时间）
    st.markdown(
        f"""
        <div style="text-align:right; color:#64748b; font-size:0.75rem; margin-top:5px;">
            数据更新时间：<span style="color:#34d399; font-weight:bold;">
            {now_cn().strftime('%Y-%m-%d %H:%M:%S')}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# 获取数据
with st.spinner('正在从华尔街连线 (Yahoo Finance)...'):
    data = get_market_data()

if data is None:
    st.error("无法获取数据，请检查网络连接（需要能访问 Yahoo Finance）。")
else:
    # 计算得分
    scores, total_score = calculate_score(data)

    # ========= 风险上限：极端环境直接给总分加“天花板” =========
    total_score_raw = total_score  # 保存原始总分

    vix = data['vix']
    us10y = data['us10y']
    dxy = data['dxy']

    # 宏观极端风险条件：高波动 / 高利率 / 极强美元（三者任一满足即视为极端环境）
    macro_hard_risk = (vix >= 30) or (us10y >= 5.0) or (dxy >= 107)

    # 如果环境极端且总分偏高，则把总分压到防守区上限（比如 55 分）
    if macro_hard_risk and total_score > 55:
        total_score = 55

    # ====== 市场状态 & 宏观风险标记（给推荐逻辑用） ======
    p, ma20, ma60 = data['price'], data['ma20'], data['ma60']

    if ma20 > ma60 * 1.01:
        regime_text = "上升趋势（偏牛市）"
    elif ma20 < ma60 * 0.99:
        regime_text = "下降趋势（偏熊市）"
    else:
        regime_text = "震荡区间"

    # 宏观高风险：高波动 / 高利率 / 极强美元，则整体建议往防守降一级
    high_risk = (vix >= 28) or (us10y >= 4.8) or (dxy >= 106)

    # ====== 基于总分的基础档位（0~3 档），再叠加宏观风险修正 ======
    if total_score >= 70:
        level = 3  # 进攻区间（偏多）
    elif total_score >= 55:
        level = 2  # 偏积极
    elif total_score >= 40:
        level = 1  # 中性偏防守
    else:
        level = 0  # 明显防守

    # 宏观压力大时，整体建议自动下调一档
    if high_risk and level > 0:
        level -= 1

    # ====== 按档位给出 A / C 不同的操作建议 ======
    risk_note = ""
    if high_risk:
        risk_note = (
            "<br><span style='font-size:0.8rem; opacity:0.8;'>"
            "⚠ 当前波动率或宏观压力偏高，系统已自动下调一级建议，更偏向防守。"
            "</span>"
        )

    if level == 3:
        # 高分 + 宏观压力可控 → 进攻区
        rec_class = "rec-success"
        rec_title = "🚀 进攻区间（偏多）"
        rec_msg = (
            f"综合得分高，当前环境属于<strong>进攻区间</strong>，大盘处于 {regime_text}。<br>"
            "• <strong>A 类</strong>：建议建立或维持中高仓位，分批加仓为主，适合 1–3 年持有；大幅回撤时可逆势加仓。<br>"
            "• <strong>C 类</strong>：允许小〜中等仓位参与波段，严格设定止盈止损，避免满仓梭哈。"
            f"{risk_note}"
        )
    elif level == 2:
        # 中高分 → 以持有/轻仓进攻为主
        rec_class = "rec-info"
        rec_title = "👌 均衡偏多（持有为主）"
        rec_msg = (
            f"综合得分偏高，市场环境整体友好，当前大盘处于 {regime_text}。<br>"
            "• <strong>A 类</strong>：建议保留/建立底仓至中等仓位，回调时逐步加仓，避免一次性重仓。<br>"
            "• <strong>C 类</strong>：以轻仓波段为主，可在回调后试探性加仓，高位适当减仓锁定收益。"
            f"{risk_note}"
        )
    elif level == 1:
        # 中等分 → 以防守、控制仓位为主
        rec_class = "rec-warning"
        rec_title = "⚠ 防守均衡（控制仓位）"
        rec_msg = (
            f"综合得分一般，环境偏震荡或宏观压力不低，当前大盘处于 {regime_text}。<br>"
            "• <strong>A 类</strong>：建议仅保留核心长期底仓，不建议大幅加仓，可以逢高适度减仓。<br>"
            "• <strong>C 类</strong>：以观望为主，仅在极端情绪/超跌时小仓短线参与，更多考虑止盈/止损而非加仓。"
            f"{risk_note}"
        )
    else:
        # 低分 → 明显防守区
        rec_class = "rec-error"
        rec_title = "🛑 高风险防守区（以减仓/观望为主）"
        rec_msg = (
            f"综合得分偏低，环境整体不利于进攻，当前大盘处于 {regime_text}。<br>"
            "• <strong>A 类</strong>：建议将仓位降到较低水平，只保留你长期最有信心的那部分底仓，或阶段性清仓观望。<br>"
            "• <strong>C 类</strong>：原则上不建议持仓，已有仓位以减仓/止损/逢高退出为主，不做逆势抄底。"
            f"{risk_note}"
        )

    # --- 仪表盘区域 ---
    st.markdown("---")
    
    # 第一行：总分与建议 + 价格走势图
    col1, col2 = st.columns([1.5, 2.5])
    
    with col1:
        # 总分展示
        st.markdown(f"""
        <div style="background:#1e293b; border-radius:16px; padding:20px; text-align:center; height:100%; border:1px solid #334155;">
            <div style="color:#64748b; font-size:0.9rem; letter-spacing:1px; margin-bottom:10px;">量化总分</div>
            <div style="font-size:4rem; font-weight:900; color:{'#34d399' if total_score > 50 else '#f43f5e'}; text-shadow: 0 0 20px rgba(255,255,255,0.1);">
                {total_score:.1f}
            </div>
            <div style="color:#475569; font-size:0.8rem;">满分 100</div>
            <div class="{rec_class}" style="margin-top:20px; text-align:left;">
                <div style="font-weight:bold; font-size:1.1rem; margin-bottom:5px;">{rec_title}</div>
                <div style="font-size:0.9rem; opacity:0.9;">{rec_msg}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # 绘制价格走势图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['history'].index, y=data['history']['Close'],
            mode='lines', name='NDX',
            line=dict(color='#0ea5e9', width=2),
            fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.1)'
        ))
        fig.update_layout(
            title={'text': f"NDX 纳指走势 (当前: {data['price']:.2f})", 'font': {'color': '#e2e8f0'}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
            xaxis=dict(showgrid=False, color='#64748b'),
            yaxis=dict(showgrid=True, gridcolor='#334155', color='#64748b')
        )
        st.plotly_chart(fig, use_container_width=True)

    # 第二行：核心指标矩阵
    st.markdown('<h3 style="margin-top:30px; color:#e2e8f0; font-size:1.2rem;">📊 因子分析矩阵</h3>', unsafe_allow_html=True)
    
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    with r1_c1:
        render_card("市盈率 PE", f"{data['pe']:.2f}", "QQQ TTM", scores['pe'], 25)
    with r1_c2:
        render_card("回撤幅度", f"-{data['drawdown']:.2f}%", "相比252日高点", scores['dd'], 20)
    with r1_c3:
        render_card("RSI (14)", f"{data['rsi']:.1f}", "相对强弱指标", scores['rsi'], 7)
    with r1_c4:
        render_card("恐慌指数 VIX", f"{data['vix']:.2f}", "波动率", scores['vix'], 8)
        
    # 第三行：宏观指标
    st.markdown('<div style="margin-top:15px;"></div>', unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3, _ = st.columns([1, 1, 1, 1])
    with r2_c1:
        render_card("10年美债收益率", f"{data['us10y']:.2f}%", "无风险利率", scores['bond'], 10)
    with r2_c2:
        render_card("美元指数 DXY", f"{data['dxy']:.2f}", "美元强度", scores['dxy'], 10)
    with r2_c3:
        render_card("趋势得分", f"{scores['trend'][0]:.2f}", "MA20/MA60 位置", scores['trend'], 20)

    # 底部说明
    st.markdown("""
    <div style="margin-top:30px; padding:15px; border:1px solid #1e40af; background:rgba(30, 64, 175, 0.1); border-radius:8px; color:#93c5fd; font-size:0.85rem;">
        <strong>💡 基金 A/C 类操作指南：</strong><br>
        • <strong>A类 (前端收费)</strong>：适合长期持有 (>2年)，管理费通常较低。当系统提示“买入/持有”时优先考虑。<br>
        • <strong>C类 (销售服务费)</strong>：适合短期波段 (<1年)，买卖灵活但持有成本随时间增加。适合“抄底”或博反弹。
    </div>
    """, unsafe_allow_html=True)

    # ========= 调试面板：查看各因子得分与风险状态 =========
    with st.expander("调试面板：因子得分与风险状态", expanded=False):
        # 分数信息
        st.write(f"原始总分（未加风险上限）：{total_score_raw:.1f}")
        st.write(f"当前总分（应用风险上限后）：{total_score:.1f}")

        # 风险上限是否生效
        if macro_hard_risk:
            st.write("⚠ 宏观极端风险条件已触发：")
            st.write(f"- VIX = {vix:.2f}（≥30 视为高波动）" if vix >= 30 else f"- VIX = {vix:.2f}")
            st.write(f"- 10年美债收益率 = {us10y:.2f}%（≥5% 视为高利率）" if us10y >= 5.0 else f"- 10年美债收益率 = {us10y:.2f}%")
            st.write(f"- 美元指数 DXY = {dxy:.2f}（≥107 视为极强美元）" if dxy >= 107 else f"- 美元指数 DXY = {dxy:.2f}")
        else:
            st.write("✅ 宏观极端风险条件未触发，风险上限未生效。")

        st.write("---")
        st.write("各因子详细得分：")
        for name, (score_k, status_k, _, _) in scores.items():
            st.write(f"- {name}: {score_k:.2f} 分（{status_k}）")
