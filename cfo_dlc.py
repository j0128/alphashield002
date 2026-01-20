# cfo_dlc.py
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AlphaStrategyDLC:
    def __init__(self):
        self.version = "3.7 (Macro + Kelly + Delist Mode)"

    # ==========================================
    # 📡 宏觀數據 (FRED + Market)
    # ==========================================
    def get_macro_regime(self, fred_key):
        """
        獲取宏觀狀態：
        1. 利率 (FRED DGS10)
        2. 銅金比 (Copper/Gold) - 經濟體溫
        3. VIX - 恐慌指數
        """
        regime = {"status": "Neutral", "score": 0, "details": {}}
        
        # 1. 抓取 FRED 10年美債
        try:
            if fred_key:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={fred_key}&file_type=json&sort_order=desc&limit=1"
                r = requests.get(url, timeout=3).json()
                rate = float(r['observations'][0]['value'])
            else:
                rate = 4.0 # Fallback
        except:
            rate = 4.0 
            
        # 2. 抓取市場數據 (銅, 金, VIX)
        try:
            # HG=F (銅), GC=F (金), ^VIX
            data = yf.download(['HG=F', 'GC=F', '^VIX'], period="5d", progress=False)['Close']
            
            # 處理 MultiIndex (新版 yfinance 可能回傳多層索引)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            copper = data.get('HG=F', pd.Series([0])).iloc[-1]
            gold = data.get('GC=F', pd.Series([1])).iloc[-1]
            vix = data.get('^VIX', pd.Series([20])).iloc[-1]
            
            cg_ratio = copper / gold if gold > 0 else 0
        except:
            copper, gold, cg_ratio, vix = 0, 0, 0, 20
            
        # 3. 綜合判定 (簡易計分卡)
        score = 0
        if rate < 4.5: score += 1      # 利率低於 4.5% = +1
        if vix < 20: score += 1        # VIX 平穩 = +1
        if cg_ratio > 0.0018: score += 1 # 銅金比上升 = +1
        
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
    # 🧮 凱利公式 (含宏觀加權)
    # ==========================================
    def calculate_kelly_fraction(self, price_series, macro_score=3):
        if len(price_series) < 30: return 0.0
        
        returns = price_series.pct_change().dropna().tail(252) # 過去一年
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
        
        # 原始 Kelly
        f = p - (q / b)
        
        # 安全係數 (Half Kelly)
        base_kelly = max(0.0, min(f * 0.5, 0.4))
        
        # 🔥 宏觀加權 (Money Flow Adjustment)
        # Score 3 (Risk On) -> 100% Kelly
        # Score 0 (Risk Off) -> 50% Kelly (保守)
        multiplier = 0.5 + (macro_score / 6) # 範圍 0.5 ~ 1.0
        
        return base_kelly * multiplier

    # ==========================================
    # ⏳ 驗屍官回測引擎 (含下架模擬)
    # ==========================================
    def run_advanced_backtest(self, targets_info, monthly_budget, start_date, end_date):
        """
        targets_info: dict, key=ticker, value={'delist_date': 'YYYY-MM-DD'}
        """
        targets = list(targets_info.keys())
        macro_tickers = ['^VIX', '^TNX']
        all_tickers = targets + macro_tickers
        
        # 1. 下載數據
        # 注意：對於已下架股票，Yahoo 可能回傳空值，我們會用 last_known_prices 模擬歸零過程
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
        
        if 'Close' in data.columns: closes = data['Close']
        else: closes = data
        
        # 處理 MultiIndex
        if isinstance(closes.columns, pd.MultiIndex):
            closes.columns = closes.columns.get_level_values(0)

        # 簡單填充 (Forward Fill) 避免假日空值
        closes = closes.ffill()

        # 宏觀數據提取
        try:
            tnx = closes.get('^TNX', pd.Series(4.0, index=closes.index))
            vix = closes.get('^VIX', pd.Series(20.0, index=closes.index))
        except:
            return None # 關鍵數據缺失
            
        # 2. 變數初始化
        portfolio_history = []
        agent_holdings = {t: 0.0 for t in targets}
        agent_cash = 0.0
        dca_holdings = {t: 0.0 for t in targets}
        total_invested = 0.0
        
        # 價格記憶體 (修復數據缺失導致的歸零 Bug)
        last_known_prices = {t: 0.0 for t in targets}
        
        valid_dates = closes.index
        if len(valid_dates) < 30: return None

        # 發薪日邏輯
        df_cal = pd.DataFrame(index=valid_dates)
        df_cal['YM'] = df_cal.index.to_period('M')
        df_reset = df_cal.reset_index()
        date_col = df_reset.columns[0]
        salary_dates = df_reset.groupby('YM')[date_col].first().values
        
        alloc_per_asset = monthly_budget / len(targets)

        # 3. 逐日模擬
        for date in valid_dates:
            current_date_str = date.strftime('%Y-%m-%d')
            
            # 宏觀狀態判定 (當日)
            d_rate = tnx.loc[date] if date in tnx.index else 4.0
            d_vix = vix.loc[date] if date in vix.index else 20.0
            macro_score = 0
            if d_rate < 4.5: macro_score += 1
            if d_vix < 20: macro_score += 1
            if d_vix < 15: macro_score += 1 # 簡化
            
            # --- A. 更新價格 (Mark to Market) ---
            daily_agent = agent_cash
            daily_dca = 0.0
            current_prices = {}
            
            for t in targets:
                # 檢查下架
                delist_date = targets_info[t].get('delist_date')
                is_dead = False
                if delist_date and current_date_str >= delist_date:
                    is_dead = True
                    p = 0.0 # 強制歸零
                else:
                    try:
                        p = float(closes.loc[date, t])
                        if np.isnan(p) or p <= 0: 
                            p = last_known_prices[t] # 使用記憶價格
                        else: 
                            last_known_prices[t] = p # 更新記憶
                    except: 
                        p = last_known_prices[t]
                
                current_prices[t] = p
                
                # 如果下架，持倉價值歸零
                if is_dead:
                    agent_holdings[t] = 0
                    dca_holdings[t] = 0
                
                if p > 0:
                    daily_agent += agent_holdings[t] * p
                    daily_dca += dca_holdings[t] * p
            
            portfolio_history.append({
                "Date": date,
                "Agent": daily_agent,
                "DCA": daily_dca,
                "Cost": total_invested
            })
            
            # --- B. 發薪日入金 ---
            if date in salary_dates:
                agent_cash += monthly_budget
                total_invested += monthly_budget
                for t in targets:
                    # 只有沒死且有價格才買入
                    delist_date = targets_info[t].get('delist_date')
                    is_dead_now = delist_date and current_date_str >= delist_date
                    
                    if not is_dead_now and current_prices.get(t, 0) > 0:
                        dca_holdings[t] += alloc_per_asset / current_prices[t]

            # --- C. Agent 策略 (含止損 & Kelly) ---
            # 隨機打散交易順序
            shuffled = list(targets)
            np.random.shuffle(shuffled)
            
            for t in shuffled:
                # 若已下市，跳過
                delist_date = targets_info[t].get('delist_date')
                if delist_date and current_date_str >= delist_date: continue

                p = current_prices.get(t, 0)
                if p <= 0: continue
                
                # 計算歷史年線
                past = closes[t].loc[:date]
                if len(past) < 201: continue
                ma200 = past.iloc[-201:-1].mean()
                if pd.isna(ma200) or ma200 == 0: continue
                
                # 止損邏輯 (Stop Loss)
                is_uptrend = p > ma200
                is_panic = d_vix > 30 # 恐慌時不輕易止損，反而可能買進
                
                # 持有中 且 跌破年線 且 無恐慌 -> 賣出 (止損)
                if (not is_uptrend) and (not is_panic) and (agent_holdings[t] > 0):
                    sell_val = agent_holdings[t] * p
                    agent_cash += sell_val
                    agent_holdings[t] = 0.0
                
                # 買入邏輯 (Buy)
                elif (is_uptrend or is_panic) and agent_cash > 100:
                    # 計算 Kelly (含宏觀加權)
                    kelly = self.calculate_kelly_fraction(past.iloc[-252:], macro_score)
                    amt = agent_cash * kelly
                    
                    if amt > 500: # 最小交易金額
                        agent_holdings[t] += amt / p
                        agent_cash -= amt

        # 4. 結算
        df_hist = pd.DataFrame(portfolio_history).set_index("Date")
        if df_hist.empty: return None
        
        final_agent = df_hist['Agent'].iloc[-1]
        final_dca = df_hist['DCA'].iloc[-1]
        years = (df_hist.index[-1] - df_hist.index[0]).days / 365.25
        if years <= 0: years = 0.01
        
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