import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# === 匯入 DLC ===
try:
    from cfo_dlc import AlphaStrategyDLC
    dlc = AlphaStrategyDLC()
    dlc_loaded = True
except:
    dlc_loaded = False

st.set_page_config(page_title="CFO 3.3 - Stable", page_icon="🦅", layout="wide")

# ==========================================
# 📡 基礎數據
# ==========================================
@st.cache_data(ttl=3600)
def get_market_data():
    try:
        vix = yf.download(['^VIX'], period="5d", progress=False)['Close'].iloc[-1]
        tnx = yf.download(['^TNX'], period="5d", progress=False)['Close'].iloc[-1]
        return float(vix), float(tnx)
    except: return 20.0, 4.0

# ==========================================
# 💻 UI
# ==========================================
st.title("🦅 CFO 3.3 - Alpha Shield (Stable)")
st.caption(f"Engine: {dlc.version if dlc_loaded else 'Error'}")

with st.sidebar:
    st.header("📂 設定")
    uploaded_file = st.file_uploader("匯入資產 CSV", type=['csv'])
    if uploaded_file:
        df_up = pd.read_csv(uploaded_file)
        cols = {c.lower(): c for c in df_up.columns}
        code_col = next((cols[c] for c in cols if c in ['code', 'ticker', '標的']), None)
        my_targets = df_up[code_col].tolist() if code_col else []
    else:
        def_targets = "NVDA, AMD, CLS, URA, LTL, META, BTC-USD, SOL-USD"
        user_in = st.text_area("代號輸入", def_targets)
        my_targets = [x.strip() for x in user_in.split(',') if x.strip()]

    current_cash = st.number_input("💰 閒置現金", value=0, step=1000)
    monthly_budget = st.number_input("💵 每月預算", value=60000, step=5000)

tab1, tab2 = st.tabs(["📡 智能診斷", "⏳ 穩定回測"])

# Tab 1: 診斷
with tab1:
    vix, tnx = get_market_data()
    c1, c2 = st.columns(2)
    c1.metric("VIX", f"{vix:.2f}")
    c2.metric("10Y Bond", f"{tnx:.2f}%")
    
    if st.button("🚀 執行診斷"):
        if not dlc_loaded: st.error("DLC Error"); st.stop()
        
        with st.spinner("計算 Kelly 倉位..."):
            data = yf.download(my_targets, period="1y", progress=False)['Close']
            report = []
            pool = monthly_budget + current_cash
            
            for t in my_targets:
                try:
                    prices = data[t].dropna()
                    if prices.empty: continue
                    curr = prices.iloc[-1]
                    ma200 = prices.rolling(200).mean().iloc[-1]
                    
                    status = "✅ BUY" if curr > ma200 else "🛡️ STOP"
                    if vix > 30 and curr < ma200: status = "🔥 SNIPER"
                    
                    kelly = 0.0
                    amt = 0
                    if "BUY" in status or "SNIPER" in status:
                        kelly = dlc.calculate_kelly_fraction(prices)
                        amt = pool * kelly
                    
                    report.append({
                        "標的": t, "現價": curr, "年線": ma200, "狀態": status,
                        "Kelly%": f"{kelly*100:.1f}%", "建議": amt
                    })
                except: pass
            
            df_rep = pd.DataFrame(report)
            st.dataframe(
                df_rep.style.format({"現價":"{:.2f}", "年線":"{:.2f}", "建議":"${:,.0f}"})
                .map(lambda x: 'color: green' if 'BUY' in x else 'color: red', subset=['狀態']),
                use_container_width=True
            )
            
            tot = df_rep['建議'].sum() if not df_rep.empty else 0
            st.success(f"💰 建議總投入: ${tot:,.0f}")
            if tot < pool: st.info(f"🛡️ 保留現金: ${pool - tot:,.0f}")

# Tab 2: 回測
with tab2:
    st.subheader("⏳ 歷史回測 (Matplotlib)")
    yrs = st.slider("年數", 3, 10, 5)
    start = f"{datetime.now().year - yrs}-01-01"
    
    if st.button("▶️ 啟動回測"):
        if not dlc_loaded: st.stop()
        with st.spinner("模擬中..."):
            res = dlc.run_advanced_backtest(my_targets, monthly_budget, start)
            
            if res:
                m = res['metrics']
                df = res['history']
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Agent 終值", f"${m['agent_final']:,.0f}", f"{m['agent_cagr']*100:.1f}%")
                c2.metric("DCA 終值", f"${m['dca_final']:,.0f}", f"{m['dca_cagr']*100:.1f}%")
                c3.metric("本金", f"${m['cost']:,.0f}")
                
                # 🔥 改用 Matplotlib 畫圖 (最穩定方案)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df['Agent'], label='Agent Strategy', color='green', linewidth=2)
                ax.plot(df.index, df['DCA'], label='DCA (Buy & Hold)', color='red', linestyle='--', alpha=0.7)
                ax.set_title("Portfolio Growth: Agent vs DCA")
                ax.set_ylabel("Total Value (USD)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig) # 顯示靜態圖
                
            else:
                st.error("❌ 回測無數據，請檢查代號或縮短年數")