import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

# === 🌟 匯入 DLC 模組 ===
# 只要 cfo_dlc.py 在同一個資料夾，這行就會生效
try:
    from cfo_dlc import AlphaStrategyDLC
    dlc = AlphaStrategyDLC()
    dlc_loaded = True
except ImportError:
    dlc_loaded = False
    st.error("❌ 找不到 cfo_dlc.py！請確認策略檔案已建立。")

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(page_title="CFO 3.1 - Kelly Agent", page_icon="🦅", layout="wide")

# ==========================================
# 📡 基礎數據函數 (保留在主程式以維持基本運作)
# ==========================================
@st.cache_data(ttl=3600)
def get_basic_market_data():
    try:
        vix = yf.download(['^VIX'], period="5d", progress=False)['Close'].iloc[-1]
        tnx = yf.download(['^TNX'], period="5d", progress=False)['Close'].iloc[-1]
        return float(vix), float(tnx)
    except: return 20.0, 4.0

# ==========================================
# 💻 UI 主程式
# ==========================================
st.title("🦅 CFO 3.1 - Alpha Shield (Kelly Edition)")
st.caption(f"Strategy Core: {dlc.version if dlc_loaded else 'N/A'} | DLC Mode: Active")

with st.sidebar:
    st.header("📂 資產配置")
    
    # 1. 資產輸入
    uploaded_file = st.file_uploader("匯入資產 CSV (Code, Value)", type=['csv'])
    if uploaded_file:
        df_up = pd.read_csv(uploaded_file)
        # 簡單處理欄位
        cols = {c.lower(): c for c in df_up.columns}
        code_col = next((cols[c] for c in cols if c in ['code', 'ticker', '標的']), None)
        my_targets = df_up[code_col].tolist() if code_col else []
    else:
        def_targets = "NVDA, AMD, CLS, URA, LTL, META, BTC-USD, SOL-USD"
        user_in = st.text_area("手動輸入代號 (逗號分隔)", def_targets)
        my_targets = [x.strip() for x in user_in.split(',') if x.strip()]

    # 2. 資金參數
    current_cash = st.number_input("💰 手上閒置現金 (Dry Powder)", value=0, step=1000)
    monthly_budget = st.number_input("💵 每月投入預算", value=60000, step=5000)
    
    st.divider()
    st.info("💡 凱利公式將根據「手上現金」計算最佳買入比例。")

# --- 頁籤 ---
tab1, tab2 = st.tabs(["📡 智能診斷 (Kelly Buy)", "⏳ 動態回測 (Visualized)"])

# Tab 1: 診斷與凱利建議
with tab1:
    vix, tnx = get_basic_market_data()
    col1, col2 = st.columns(2)
    col1.metric("VIX 恐慌指數", f"{vix:.2f}")
    col2.metric("10年美債", f"{tnx:.2f}%")
    
    if st.button("🚀 執行 Agent 診斷 + 凱利計算"):
        if not dlc_loaded:
            st.stop()
            
        with st.spinner("正在計算波動率與最佳倉位..."):
            # 下載數據進行即時運算
            data = yf.download(my_targets, period="1y", progress=False)['Close']
            
            report = []
            for t in my_targets:
                try:
                    prices = data[t].dropna()
                    curr = prices.iloc[-1]
                    ma200 = prices.rolling(200).mean().iloc[-1]
                    
                    # 1. 判斷多空
                    status = "✅ BUY" if curr > ma200 else "🛡️ STOP"
                    if vix > 30 and curr < ma200: status = "🔥 SNIPER"
                    
                    # 2. 計算 Kelly 建議 (只針對 BUY/SNIPER 計算)
                    kelly_pct = 0.0
                    buy_amt = 0
                    note = "觀望"
                    
                    if "BUY" in status or "SNIPER" in status:
                        kelly_pct = dlc.calculate_kelly_fraction(prices)
                        # 資金池 = 本月薪水 + 手上現金
                        total_pool = monthly_budget + current_cash
                        # 為避免單一標的過重，將資金池除以標的數做分母
                        # 但 Kelly 是針對「勝率」下注，所以這裡我們用 (總池 * Kelly)
                        buy_amt = total_pool * kelly_pct
                        note = f"建議投入 {kelly_pct*100:.1f}% 現金"
                    
                    report.append({
                        "標的": t,
                        "現價": curr,
                        "年線": ma200,
                        "狀態": status,
                        "Kelly比例": f"{kelly_pct*100:.1f}%",
                        "建議買入金額": buy_amt
                    })
                except Exception as e:
                    pass
            
            # 顯示結果
            df_rep = pd.DataFrame(report)
            st.dataframe(
                df_rep.style.format({"現價": "{:.2f}", "年線": "{:.2f}", "建議買入金額": "${:,.0f}"})
                .map(lambda x: 'color: green' if 'BUY' in x else 'color: red', subset=['狀態']),
                use_container_width=True
            )
            
            total_buy = df_rep['建議買入金額'].sum()
            st.success(f"💰 本月凱利公式建議總買入額: ${total_buy:,.0f}")
            if total_buy < (monthly_budget + current_cash):
                st.warning(f"🛡️ 剩餘現金 ${ (monthly_budget + current_cash) - total_buy:,.0f} 請留存至現金池 (Dry Powder)。")

# Tab 2: 回測
with tab2:
    st.subheader("⏳ 歷史回測 (含資產曲線)")
    backtest_years = st.slider("回測年數", 3, 10, 5)
    start_year = datetime.now().year - backtest_years
    start_date = f"{start_year}-01-01"
    
    if st.button("▶️ 啟動時光機"):
        if not dlc_loaded: st.stop()
        
        with st.spinner(f"正在模擬 {start_date} 至今的交易..."):
            res = dlc.run_advanced_backtest(my_targets, monthly_budget, start_date)
            
            metrics = res['metrics']
            df_hist = res['history_df']
            
            # 1. 顯示關鍵指標
            m1, m2, m3 = st.columns(3)
            m1.metric("Agent 最終資產", f"${metrics['agent_final']:,.0f}", f"年化 {metrics['agent_cagr']*100:.1f}%")
            m2.metric("DCA (笨定投) 資產", f"${metrics['dca_final']:,.0f}", f"年化 {metrics['dca_cagr']*100:.1f}%")
            m3.metric("總投入本金", f"${metrics['total_invested']:,.0f}")
            
            # 2. 畫圖 (解決之前跑不出來的問題)
            st.markdown("### 📈 資產成長曲線")
            # 只畫 Agent 和 DCA 的權益曲線
            chart_data = df_hist[['Agent_Equity', 'DCA_Equity']]
            st.line_chart(chart_data, color=["#00FF00", "#FF0000"])
            
            # 3. 顯示詳細數據
            with st.expander("查看詳細每日數據"):
                st.dataframe(df_hist)