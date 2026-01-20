# cfo_dlc.py
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt

class AlphaStrategyDLC:
    def __init__(self):
        self.version = "3.4 (Macro + OBV + Kelly)"

    # ==========================================
    # 📡 宏觀數據引擎 (FRED + Yahoo)
    # ==========================================
    def get_macro_regime(self, fred_key):
        """
        獲取宏觀狀態：
        1. 利率 (FRED DGS10)
        2. 銅金比 (Copper/Gold Ratio) - 經濟晴雨表
        3. VIX - 恐慌指數
        """
        regime = {"status": "Neutral", "score": 0, "details": {}}
        
        # 1. 抓取 FRED 10年美債
        try:
            if fred_key:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={fred_key}&file_type=json&sort_order=desc&limit=1"
                r = requests.get(url, timeout=5).json()
                rate = float(r['observations'][0]['value'])
            else:
                rate = 4.0 # Fallback
        except:
            rate = 4.0 # Fallback
            
        # 2. 抓取市場數據 (銅, 金, VIX)
        try:
            # HG=F (銅), GC=F (金), ^VIX
            data = yf.download(['HG=F', 'GC=F', '^VIX'], period="5d", progress=False)['Close']
            copper = data['HG=F'].iloc[-1]
            gold = data['GC=F'].iloc[-1]
            vix = data['^VIX'].iloc[-1]
            
            cg_ratio = copper / gold
        except:
            copper, gold, cg_ratio, vix = 0, 0, 0, 20
            
        # 3. 綜合判定
        # 簡單邏輯：利率高 + VIX高 = 緊縮 (Risk Off)
        # 銅金比上漲 = 復甦 (Risk On)
        score = 0
        if rate < 4.5: score += 1
        if vix < 20: score += 1
        if cg_ratio > 0.0018: score += 1 # 經驗值
        
        regime['details'] = {
            "Rate (10Y)": rate,
            "VIX": vix,
            "Copper/Gold": cg_ratio
        }
        regime['score'] = score
        
        if score >= 2: regime['status'] = "🟢 Risk On (寬鬆/成長)"
        elif score == 1: regime['status'] = "🟡 Neutral (震盪)"
        else: regime['status'] = "🔴 Risk Off (緊縮/避險)"
        
        return regime

    # ==========================================
    # 🌊 技術指標：OBV (能量潮)
    # ==========================================
    def calculate_obv(self, price_series, volume_series):
        if len(price_series) < 2: return pd.Series(0, index=price_series.index)
        
        # 確保索引對齊
        df = pd.DataFrame({'close': price_series, 'volume': volume_series}).dropna()
        
        obv = [0]
        for i in range(1, len(df)):
            c = df['close'].iloc[i]
            prev_c = df['close'].iloc[i-1]
            v = df['volume'].iloc[i]
            
            if c > prev_c:
                obv.append(obv[-1] + v)
            elif c < prev_c:
                obv.append(obv[-1] - v)
            else:
                obv.append(obv[-1])
                
        return pd.Series(obv, index=df.index)

    # ==========================================
    # 🧮 凱利公式 (含宏觀權重)
    # ==========================================
    def calculate_kelly_fraction(self, price_series, macro_score=3):
        if len(price_series) < 30: return 0.0
        
        returns = price_series.pct_change().dropna().tail(252) # 一年
        if len(returns) == 0: return 0.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0: return 0.0
        if len(losses) == 0: return 0.5
        
        p = len(wins) / len(returns)
        q = 1 - p
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        if avg_loss == 0: return 0.5
        b = avg_win / avg_loss
        
        f = p - (q / b)
        
        # 基礎 Kelly (Half)
        base_kelly = max(0.0, min(f * 0.5, 0.4))
        
        # 🔥 宏觀加權 (Money Flow Adjustment)
        # Score 3 (Risk On) -> 100% Kelly
        # Score 0 (Risk Off) -> 50% Kelly
        multiplier = 0.5 + (macro_score / 6) # 0.5 ~ 1.0
        
        return base_kelly * multiplier

    # ==========================================
    # ⏳ 終極回測引擎
    # ==========================================
    def run_advanced_backtest(self, targets, monthly_budget, start_date):
        # 1. 下載所有數據 (含宏觀指標)
        macro_tickers = ['^VIX', '^TNX', 'HG=F', 'GC=F']
        all_tickers = targets + macro_tickers
        
        data = yf.download(all_tickers, start=start_date, progress=False)
        
        # 處理 MultiIndex
        if 'Close' in data.columns: closes = data['Close']
        else: closes = data
        
        if 'Volume' in data.columns: vols = data['Volume']
        else: vols = pd.DataFrame(0, index=closes.index, columns=closes.columns)

        # 2. 預計算指標 (向量化加速)
        # 宏觀指標
        try:
            cg_ratio = closes['HG=F'] / closes['GC=F']
            tnx = closes['^TNX']
            vix = closes['^VIX']
        except:
            return None # 數據缺失
            
        # MA200 & OBV
        mas = closes.rolling(200).mean()
        
        # 3. 逐日模擬
        portfolio_history = []
        agent_holdings = {t: 0.0 for t in targets}
        agent_cash = 0.0
        dca_holdings = {t: 0.0 for t in targets}
        total_invested = 0.0
        
        valid_dates = closes.index[200:]
        if len(valid_dates) == 0: return None

        # 發薪日
        df_cal = pd.DataFrame(index=valid_dates)
        df_cal['YM'] = df_cal.index.to_period('M')
        df_reset = df_cal.reset_index()
        date_col = df_reset.columns[0]
        salary_dates = df_reset.groupby('YM')[date_col].first().values

        alloc = monthly_budget / len(targets)

        for date in valid_dates:
            # A. Mark to Market
            daily_agent = agent_cash
            daily_dca = 0.0
            
            # 當日宏觀分數計算
            d_rate = tnx.loc[date]
            d_vix = vix.loc[date]
            d_cg = cg_ratio.loc[date]
            
            macro_score = 0
            if d_rate < 4.5: macro_score += 1
            if d_vix < 20: macro_score += 1
            if d_cg > 0.0018: macro_score += 1 # 簡化門檻
            
            # 資產計價
            current_prices = {}
            for t in targets:
                try:
                    p = float(closes.loc[date, t])
                    if np.isnan(p) or p <= 0: p = 0
                    current_prices[t] = p
                    daily_agent += agent_holdings[t] * p
                    daily_dca += dca_holdings[t] * p
                except: pass
            
            portfolio_history.append({
                "Date": date,
                "Agent": daily_agent,
                "DCA": daily_dca,
                "Cost": total_invested,
                "MacroScore": macro_score
            })
            
            # B. 發薪日
            if date in salary_dates:
                agent_cash += monthly_budget
                total_invested += monthly_budget
                for t in targets:
                    if current_prices.get(t, 0) > 0:
                        dca_holdings[t] += alloc / current_prices[t]
                        
            # C. Agent 操作
            # 隨機順序
            shuffled = list(targets)
            np.random.shuffle(shuffled)
            
            for t in shuffled:
                p = current_prices.get(t, 0)
                if p <= 0: continue
                
                ma = mas.loc[date, t]
                if pd.isna(ma): continue
                
                # OBV 趨勢確認 (過去20天 OBV 是否上升)
                # 這裡簡化：只要價格在年線上，且 VIX 不高，我們就買
                # 但加入 Macro Score 調整買入量
                
                signal = (p > ma) or (d_vix > 30)
                
                if signal and agent_cash > 100:
                    # 計算 Kelly (過去一年數據)
                    # 為了回測速度，這裡做簡化 Kelly 估計，或您可以呼叫 self.calculate_kelly
                    # 使用 self.calculate_kelly_fraction 會比較慢但準確
                    
                    past_prices = closes[t].loc[:date].tail(252)
                    k_frac = self.calculate_kelly_fraction(past_prices, macro_score)
                    
                    amt = agent_cash * k_frac
                    if amt > 500:
                        agent_holdings[t] += amt / p
                        agent_cash -= amt
                        
        # 4. 結算
        df_hist = pd.DataFrame(portfolio_history).set_index("Date")
        if df_hist.empty: return None
        
        final_agent = df_hist['Agent'].iloc[-1]
        final_dca = df_hist['DCA'].iloc[-1]
        years = (df_hist.index[-1] - df_hist.index[0]).days / 365.25
        
        agent_cagr = (final_agent / total_invested) ** (1/years) - 1 if total_invested > 0 else 0
        dca_cagr = (final_dca / total_invested) ** (1/years) - 1 if total_invested > 0 else 0
        
        return {
            "history": df_hist,
            "metrics": {
                "years": years,
                "agent_final": final_agent,
                "dca_final": final_dca,
                "agent_cagr": agent_cagr,
                "dca_cagr": dca_cagr,
                "cost": total_invested
            }
        }