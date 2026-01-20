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

st.set_page_config(page_title="CFO 3.4 - Quant Master", page_icon="🦅", layout="wide")

# ==========================================
# 🔑 讀取 Secrets
# ==========================================
try:
    FRED_KEY = st.secrets["FRED_API_KEY"]
except:
    st.error("⚠️ 未偵測到 FRED API Key，請在 secrets.toml 中設定。")
    FRED_KEY = None

# ==========================================
# 💻 UI 主介面
# ==========================================
st.title("🦅 CFO 3.4 - 量化戰情室 (Quant Master)")
st.caption(f"Engine: {dlc.version if dlc_loaded else 'N/A'} | FRED Link: {'✅ Active' if FRED_KEY else '❌ Inactive'}")

with st.sidebar:
    st.header("📂 資產配置")
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

# --- 宏觀儀表板 ---
if dlc_loaded:
    st.subheader("🌍 全球宏觀天氣 (Macro Regime)")
    with st.spinner("正在連線 FRED 與期貨市場..."):
        regime = dlc.get_macro_regime(FRED_KEY)
        
    det = regime['details']
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("狀態", regime['status'], f"Score: {regime['score']}/3")
    m2.metric("10年美債 (FRED)", f"{det['Rate (10Y)']:.2f}%", "-緊縮" if det['Rate (10Y)']>4.5 else "寬鬆")
    m3.metric("銅金比 (景氣)", f"{det['Copper/Gold']:.4f}", "上升=復甦")
    m4.metric("VIX (恐慌)", f"{det['VIX']:.2f}", "避險" if det['VIX']>20 else "平穩")
    st.divider()

tab1, tab2 = st.tabs(["📡 深度量化診斷 (OBV + Kelly)", "⏳ 歷史回測 (含年化率)"])

# Tab 1: 診斷
with tab1:
    if st.button("🚀 執行全因子掃描"):
        if not dlc_loaded: st.stop()
        with st.spinner("正在計算 MA200, OBV, Kelly..."):
            # 下載數據 (含成交量)
            data = yf.download(my_targets, period="1y", progress=False)
            if 'Close' in data.columns: 
                closes = data['Close']
                vols = data['Volume']
            else:
                closes = data
                vols = data # 兼容性 fallback
            
            report = []
            pool = monthly_budget + current_cash
            macro_score = regime['score']
            
            for t in my_targets:
                try:
                    prices = closes[t].dropna()
                    volume = vols[t].dropna()
                    if prices.empty: continue
                    
                    curr = prices.iloc[-1]
                    ma200 = prices.rolling(200).mean().iloc[-1]
                    
                    # 計算 OBV
                    obv = dlc.calculate_obv(prices, volume)
                    obv_slope = obv.diff(20).iloc[-1] # 20天 OBV 趨勢
                    obv_sig = "↗️ 增強" if obv_slope > 0 else "↘️ 流出"
                    
                    # 狀態判定
                    status = "✅ BUY" if curr > ma200 else "🛡️ STOP"
                    if det['VIX'] > 30 and curr < ma200: status = "🔥 SNIPER"
                    
                    # Kelly 計算
                    kelly = 0.0
                    amt = 0
                    if "BUY" in status or "SNIPER" in status:
                        kelly = dlc.calculate_kelly_fraction(prices, macro_score)
                        amt = pool * kelly
                    
                    report.append({
                        "標的": t, 
                        "現價": curr, 
                        "年線": ma200, 
                        "資金流 (OBV)": obv_sig,
                        "狀態": status,
                        "Kelly% (含宏觀)": f"{kelly*100:.1f}%", 
                        "建議投入": amt
                    })
                except: pass
            
            df_rep = pd.DataFrame(report)
            st.dataframe(
                df_rep.style.format({"現價":"{:.2f}", "年線":"{:.2f}", "建議投入":"${:,.0f}"})
                .map(lambda x: 'color: green' if 'BUY' in x else 'color: red', subset=['狀態']),
                use_container_width=True
            )
            
            tot = df_rep['建議投入'].sum() if not df_rep.empty else 0
            st.success(f"💰 本月建議總投入: ${tot:,.0f}")
            if tot < pool: st.info(f"🛡️ 宏觀避險/現金保留: ${pool - tot:,.0f}")

# Tab 2: 回測
with tab2:
    st.subheader("⏳ 歷史回測 (Matplotlib)")
    yrs = st.slider("回測年數", 3, 10, 5)
    start = f"{datetime.now().year - yrs}-01-01"
    
    if st.button("▶️ 啟動回測"):
        if not dlc_loaded: st.stop()
        with st.spinner(f"正在模擬 {start} 至今的交易 (含宏觀動態權重)..."):
            res = dlc.run_advanced_backtest(my_targets, monthly_budget, start)
            
            if res:
                m = res['metrics']
                df = res['history']
                
                # 1. 顯示關鍵指標 (含 CAGR)
                c1, c2, c3 = st.columns(3)
                c1.metric("Agent 最終資產", f"${m['agent_final']:,.0f}", f"年化 (CAGR): {m['agent_cagr']*100:.2f}%")
                c2.metric("DCA 笨定投", f"${m['dca_final']:,.0f}", f"年化 (CAGR): {m['dca_cagr']*100:.2f}%")
                c3.metric("總投入本金", f"${m['cost']:,.0f}")
                
                # 2. 畫圖 (Matplotlib)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df['Agent'], label=f'Agent (CAGR {m["agent_cagr"]*100:.1f}%)', color='green', linewidth=2)
                ax.plot(df.index, df['DCA'], label=f'DCA (CAGR {m["dca_cagr"]*100:.1f}%)', color='red', linestyle='--', alpha=0.7)
                ax.plot(df.index, df['Cost'], label='Invested Capital', color='gray', linestyle=':', alpha=0.5)
                
                ax.set_title(f"Portfolio Growth ({yrs} Years) - Macro Adjusted")
                ax.set_ylabel("Total Value (USD)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
            else:
                st.error("❌ 回測無數據，請檢查代號或縮短年數")