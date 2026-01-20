# cfo_dlc.py
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class AlphaStrategyDLC:
    def __init__(self):
        self.version = "3.3 (Stable Matplotlib Edition)"

    # ==========================================
    # 🧮 凱利公式核心引擎
    # ==========================================
    def calculate_kelly_fraction(self, price_series, window=252):
        if len(price_series) < 30: return 0.0
        
        returns = price_series.pct_change().dropna().tail(window)
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
        return max(0.0, min(f * 0.5, 0.4))

    # ==========================================
    # ⏳ 進階回測引擎
    # ==========================================
    def run_advanced_backtest(self, targets, monthly_budget, start_date):
        # 1. 下載數據
        tickers = targets + ['^VIX']
        data = yf.download(tickers, start=start_date, progress=False)
        
        if 'Close' in data.columns: closes = data['Close']
        else: closes = data

        try:
            vix_series = closes['^VIX']
        except:
            vix_series = pd.Series(20, index=closes.index)

        # 2. 變數初始化
        portfolio_history = []
        agent_holdings = {t: 0.0 for t in targets}
        agent_cash = 0.0
        dca_holdings = {t: 0.0 for t in targets}
        total_invested = 0.0
        
        # 3. 日期處理 (修復版)
        valid_dates = closes.index[200:]
        if len(valid_dates) == 0: return None

        df_cal = pd.DataFrame(index=valid_dates)
        df_cal['YM'] = df_cal.index.to_period('M')
        # 動態獲取日期欄位名稱
        df_reset = df_cal.reset_index()
        date_col = df_reset.columns[0]
        salary_dates = df_reset.groupby('YM')[date_col].first().values

        alloc_per_asset = monthly_budget / len(targets)

        # 4. 模擬迴圈
        for date in valid_dates:
            # A. Mark to Market
            daily_agent = agent_cash
            daily_dca = 0.0
            current_prices = {}
            
            skip = False
            for t in targets:
                try:
                    p = float(closes.loc[date, t])
                    if np.isnan(p) or p <= 0: p = 0
                    current_prices[t] = p
                    
                    daily_agent += agent_holdings[t] * p
                    daily_dca += dca_holdings[t] * p
                except: skip = True
            
            if skip: continue

            portfolio_history.append({
                "Date": date,
                "Agent": daily_agent,
                "DCA": daily_dca,
                "Cost": total_invested
            })

            # B. 發薪日
            if date in salary_dates:
                agent_cash += monthly_budget
                total_invested += monthly_budget
                for t in targets:
                    if current_prices.get(t, 0) > 0:
                        dca_holdings[t] += alloc_per_asset / current_prices[t]

            # C. Kelly Buy
            vix = float(vix_series.loc[date])
            # 隨機順序
            shuffled = list(targets)
            np.random.shuffle(shuffled)

            for t in shuffled:
                p = current_prices.get(t, 0)
                if p <= 0: continue
                
                past = closes[t].loc[:date]
                if len(past) < 201: continue
                
                ma200 = past.iloc[-201:-1].mean()
                
                if (p > ma200 or vix > 30) and agent_cash > 100:
                    kelly = self.calculate_kelly_fraction(past.iloc[-252:])
                    amt = agent_cash * kelly
                    if amt > 500:
                        agent_holdings[t] += amt / p
                        agent_cash -= amt

        # 5. 結果整理
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