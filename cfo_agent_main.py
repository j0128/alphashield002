import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# === 🌟 匯入 DLC 模組 ===
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
st.set_page_config(page_title="CFO 3.2 - Kelly Agent", page_icon="🦅", layout="wide")

# ==========================================
# 📡 基礎數據函數
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
st.title("🦅 CFO 3.2 - Alpha Shield (Kelly Edition)")
st.caption(f"Core Engine: {dlc.version if dlc_loaded else 'N/A'}")

with st.sidebar:
    st.header("📂 資產配置")
    
    # 1. 資產輸入
    uploaded_file = st.file_uploader("匯入資產 CSV", type=['csv'])
    if uploaded_file:
        df_up = pd.read_csv(uploaded_file)
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
    st.info("💡 凱利公式將自動計算最佳倉位。")

# --- 頁籤 ---
tab1, tab2 = st.tabs(["📡 智能診斷 (Kelly Buy)", "⏳ 動態回測 (Visualized)"])

# Tab 1: 診斷
with tab1:
    vix, tnx = get_basic_market_data()
    c1, c2 = st.columns(2)
    c1.metric("VIX 恐慌指數", f"{vix:.2f}")
    c2.metric("10年美債", f"{tnx:.2f}%")
    
    if st.button("🚀 執行診斷"):
        if not dlc_loaded: st.stop()
        with st.spinner("正在計算 Kelly 最佳倉位..."):
            data = yf.download(my_targets, period="1y", progress=False)['Close']
            report = []
            
            # 計算總資金池 (薪水 + 手上現金)
            total_pool = monthly_budget + current_cash
            
            for t in my_targets:
                try:
                    prices = data[t].dropna()
                    if prices.empty: continue
                    curr = prices.iloc[-1]
                    ma200 = prices.rolling(200).mean().iloc[-1]
                    
                    status = "✅ BUY" if curr > ma200 else "🛡️ STOP"
                    if vix > 30 and curr < ma200: status = "🔥 SNIPER"
                    
                    kelly_pct = 0.0
                    buy_amt = 0
                    
                    if "BUY" in status or "SNIPER" in status:
                        kelly_pct = dlc.calculate_kelly_fraction(prices)
                        buy_amt = total_pool * kelly_pct
                    
                    report.append({
                        "標的": t, "現價": curr, "年線": ma200, "狀態": status,
                        "Kelly建議%": f"{kelly_pct*100:.1f}%",
                        "建議金額": buy_amt
                    })
                except: pass
            
            df_rep = pd.DataFrame(report)
            st.dataframe(
                df_rep.style.format({"現價":"{:.2f}", "年線":"{:.2f}", "建議金額":"${:,.0f}"})
                .map(lambda x: 'color: green' if 'BUY' in x else 'color: red', subset=['狀態']),
                use_container_width=True
            )
            
            total_buy = df_rep['建議金額'].sum() if not df_rep.empty else 0
            st.success(f"💰 本月總建議投入: ${total_buy:,.0f}")
            if total_buy < total_pool:
                st.info(f"🛡️ 應保留現金: ${total_pool - total_buy:,.0f}")

# Tab 2: 回測
with tab2:
    st.subheader("⏳ 歷史回測 (Agent vs DCA)")
    yrs = st.slider("回測年數", 3, 10, 5)
    start_date = f"{datetime.now().year - yrs}-01-01"
    
    if st.button("▶️ 啟動 Kelly 回測"):
        if not dlc_loaded: st.stop()
        with st.spinner(f"正在模擬 {start_date} 至今的交易 (含 Kelly 邏輯)..."):
            res = dlc.run_advanced_backtest(my_targets, monthly_budget, start_date)
            
            if res:
                metrics = res['metrics']
                df_hist = res['history_df']
                
                # 1. 關鍵指標區
                col1, col2, col3 = st.columns(3)
                col1.metric("Agent 最終資產", f"${metrics['agent_final']:,.0f}", f"年化 {metrics['agent_cagr']*100:.2f}%")
                col2.metric("DCA (笨定投) 資產", f"${metrics['dca_final']:,.0f}", f"年化 {metrics['dca_cagr']*100:.2f}%")
                col3.metric("總投入本金", f"${metrics['total_invested']:,.0f}")
                
                # 2. 顯示圖表
                st.markdown("### 📈 資產對決曲線")
                st.line_chart(df_hist[['Agent策略', 'DCA定投']], color=["#00FF00", "#FF4B4B"])
                
                # 3. 顯示 Alpha
                alpha = (metrics['agent_cagr'] - metrics['dca_cagr']) * 100
                if alpha > 0:
                    st.success(f"🏆 Agent 策略每年平均比笨定投多賺 **{alpha:.2f}%**")
                else:
                    st.warning(f"⚠️ Agent 策略落後 {alpha:.2f}% (可能是因為過度保守)")
            else:
                st.error("❌ 回測失敗：可能是數據不足或網路問題，請縮短年數再試。")