import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, date

# === 匯入 DLC ===
try:
    from cfo_dlc import AlphaStrategyDLC
    dlc = AlphaStrategyDLC()
    dlc_loaded = True
except:
    dlc_loaded = False

st.set_page_config(page_title="CFO 3.8 - Quant Master", page_icon="🦅", layout="wide")

# ==========================================
# 🔑 讀取 Secrets
# ==========================================
try:
    FRED_KEY = st.secrets["FRED_API_KEY"]
except:
    FRED_KEY = None 

# ==========================================
# 💻 UI 主介面
# ==========================================
st.title("🦅 CFO 3.8 - 量化戰情室 (Quant Master)")
st.caption(f"Engine: {dlc.version if dlc_loaded else 'N/A'} | Target: CAGR 30% | Stop-Loss: Active")

with st.sidebar:
    st.header("📂 實驗室設定")
    
    targets_info = {} 
    
    # --- 1. CSV 上傳區 ---
    st.subheader("1. 批次匯入 (CSV)")
    uploaded_file = st.file_uploader("上傳資產名單", type=['csv'], help="欄位: Code, DelistDate (選填)")
    
    if uploaded_file:
        df_up = pd.read_csv(uploaded_file)
        cols = {c.lower(): c for c in df_up.columns}
        
        code_col = next((cols[c] for c in cols if c in ['code', 'ticker', '標的']), None)
        delist_col = next((cols[c] for c in cols if 'delist' in c or '下架' in c), None)
        
        if code_col:
            for _, row in df_up.iterrows():
                t = str(row[code_col]).strip().upper() # 強制轉大寫
                d_date = None
                if delist_col and pd.notna(row[delist_col]):
                    d_date = str(row[delist_col]).strip()
                targets_info[t] = {'delist_date': d_date}

    # --- 2. 手動輸入區 (混合模式) ---
    st.subheader("2. 手動輸入 / 補強")
    # 如果 CSV 有東西，預設值留空；如果沒 CSV，給預設範例
    default_text = "" if targets_info else "NVDA, AMD, META, BTC-USD"
    user_in = st.text_area("輸入代號 (逗號分隔)", value=default_text, height=100)
    
    if user_in:
        manual_list = [x.strip().upper() for x in user_in.split(',') if x.strip()]
        for t in manual_list:
            # 如果 CSV 已經有這個代號，手動輸入的不會覆蓋它 (保留 CSV 的下架設定)
            if t not in targets_info:
                targets_info[t] = {'delist_date': None}

    # 防呆：如果兩者都空，給預設範例
    if not targets_info:
        def_targets = ["NVDA", "AMD", "META", "BTC-USD", "SIVB"]
        for t in def_targets:
            d = '2023-03-10' if t == 'SIVB' else None
            targets_info[t] = {'delist_date': d}

    # 顯示最終監測名單 (可摺疊)
    count = len(targets_info)
    with st.expander(f"📊 目前監測名單 ({count} 檔)"):
        st.json(targets_info)

    st.divider()
    current_cash = st.number_input("💰 閒置現金 (Dry Powder)", value=0, step=1000)
    monthly_budget = st.number_input("💵 每月投入預算", value=60000, step=5000)

# --- 宏觀儀表板 ---
if dlc_loaded:
    with st.spinner("正在連線 FRED 與期貨市場..."):
        regime = dlc.get_macro_regime(FRED_KEY)
        
    det = regime['details']
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("宏觀狀態", regime['status'], f"Score: {regime['score']}/3")
    m2.metric("10Y 美債 (FRED)", f"{det['Rate (10Y)']:.2f}%", "高於4.5%為緊縮")
    m3.metric("銅金比 (景氣)", f"{det['Copper/Gold']:.4f}", "越高越好")
    m4.metric("VIX (恐慌)", f"{det['VIX']:.2f}", "低於20為平穩")
    st.divider()

tab1, tab2 = st.tabs(["📡 深度量化診斷 (OBV + Kelly)", "⏳ 時光機回測 (驗屍模式)"])

# Tab 1: 診斷
with tab1:
    st.markdown("#### 🎯 本月戰術指令")
    if st.button("🚀 執行全因子掃描"):
        if not dlc_loaded: st.stop()
        
        # 只掃描還沒下架的
        active_targets = [t for t, info in targets_info.items() if not info.get('delist_date')]
        
        if not active_targets:
            st.warning("名單中的股票皆已下架。")
        else:
            with st.spinner("正在計算 MA200, OBV, Kelly..."):
                # 下載數據 (含成交量)
                data = yf.download(active_targets, period="1y", progress=False)
                
                # 兼容性處理
                if 'Close' in data.columns: 
                    closes = data['Close']
                    vols = data['Volume']
                else:
                    closes = data
                    vols = data # 假設無量
                
                # 處理 MultiIndex
                if isinstance(closes.columns, pd.MultiIndex):
                    closes.columns = closes.columns.get_level_values(0)
                if isinstance(vols, pd.DataFrame) and isinstance(vols.columns, pd.MultiIndex):
                    vols.columns = vols.columns.get_level_values(0)
                
                report = []
                pool = monthly_budget + current_cash
                macro_score = regime['score']
                
                for t in active_targets:
                    try:
                        # 處理單一 ticker 下載時並非 DataFrame 的情況
                        if len(active_targets) == 1:
                            prices = closes.dropna()
                            volume = vols.dropna()
                        else:
                            prices = closes[t].dropna()
                            volume = vols[t].dropna() if t in vols else pd.Series()

                        if prices.empty: continue
                        
                        curr = prices.iloc[-1]
                        ma200 = prices.rolling(200).mean().iloc[-1]
                        
                        # OBV
                        obv_sig = "N/A"
                        if not volume.empty and len(volume) == len(prices):
                            obv = dlc.calculate_obv(prices, volume)
                            if len(obv) > 20:
                                slope = obv.diff(20).iloc[-1]
                                obv_sig = "↗️ 增強" if slope > 0 else "↘️ 流出"
                        
                        # 狀態
                        status = "✅ BUY" if curr > ma200 else "🛡️ STOP"
                        if det['VIX'] > 30 and curr < ma200: status = "🔥 SNIPER"
                        
                        # Kelly
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
                            "Kelly%": f"{kelly*100:.1f}%", 
                            "建議投入": amt
                        })
                    except: pass
                
                df_rep = pd.DataFrame(report)
                if not df_rep.empty:
                    st.dataframe(
                        df_rep.style.format({"現價":"{:.2f}", "年線":"{:.2f}", "建議投入":"${:,.0f}"})
                        .map(lambda x: 'color: green' if 'BUY' in x else 'color: red', subset=['狀態']),
                        use_container_width=True
                    )
                    
                    tot = df_rep['建議投入'].sum()
                    st.success(f"💰 本月建議總投入: ${tot:,.0f}")
                    if tot < pool: st.info(f"🛡️ 宏觀避險/現金保留: ${pool - tot:,.0f}")
                else:
                    st.error("無法取得數據，請檢查代號。")

# Tab 2: 回測
with tab2:
    st.subheader("⏳ 時光機設定")
    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("開始日期", date(2020, 1, 1))
    end_date = col_d2.date_input("結束日期", date.today())
    
    if start_date >= end_date:
        st.error("開始日期必須早於結束日期")
    
    if st.button("▶️ 啟動驗屍回測"):
        if not dlc_loaded: st.stop()
        
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")
        
        with st.spinner(f"正在模擬 {s_str} 至 {e_str} 的市場 (含下架事件)..."):
            res = dlc.run_advanced_backtest(targets_info, monthly_budget, s_str, e_str)
            
            if res:
                m = res['metrics']
                df = res['history']
                
                # 關鍵指標
                c1, c2, c3 = st.columns(3)
                c1.metric("Agent 最終資產", f"${m['agent_final']:,.0f}", f"CAGR {m['agent_cagr']*100:.1f}%")
                c2.metric("DCA 笨定投", f"${m['dca_final']:,.0f}", f"CAGR {m['dca_cagr']*100:.1f}%")
                
                alpha = m['agent_final'] - m['dca_final']
                c3.metric("Alpha (避險價值)", f"${alpha:,.0f}", delta_color="normal" if alpha > 0 else "inverse")
                
                # 畫圖 (Matplotlib)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df['Agent'], label=f'Agent (CAGR {m["agent_cagr"]*100:.1f}%)', color='green', linewidth=2)
                ax.plot(df.index, df['DCA'], label=f'DCA (CAGR {m["dca_cagr"]*100:.1f}%)', color='red', linestyle='--', alpha=0.7)
                ax.plot(df.index, df['Cost'], label='Cost Basis', color='gray', linestyle=':', alpha=0.5)
                
                # 標記下架事件
                for t, info in targets_info.items():
                    d_date = info.get('delist_date')
                    if d_date and s_str <= d_date <= e_str:
                        d_dt = pd.to_datetime(d_date)
                        ax.axvline(x=d_dt, color='black', linestyle=':', alpha=0.5)
                        ax.text(d_dt, df['Agent'].max()*0.5, f" {t} DELIST", rotation=90, color='red')

                ax.set_title(f"Simulation: {s_str} ~ {e_str}")
                ax.set_ylabel("Total Value (USD)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
            else:
                st.error("回測失敗。可能是日期區間無數據，或所有股票皆已下架。")