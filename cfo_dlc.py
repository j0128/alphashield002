# cfo_dlc.py
import pandas as pd
import numpy as np
import yfinance as yf

class AlphaStrategyDLC:
    def __init__(self):
        self.version = "3.2 (Kelly Backtest + Bug Fix)"

    # ==========================================
    # 🧮 凱利公式核心引擎 (Kelly Criterion)
    # ==========================================
    def calculate_kelly_fraction(self, price_series, window=252):
        """
        計算凱利公式建議的倉位比例 (Half-Kelly)
        """
        if len(price_series) < 30: return 0.0 # 數據太少不計算
        
        # 計算日報酬率
        returns = price_series.pct_change().dropna().tail(window)
        if len(returns) == 0: return 0.0

        # 1. 勝率 (Win Rate, p)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0: return 0.0
        if len(losses) == 0: return 0.5 # 全勝，限制最大 50%
        
        p = len(wins) / len(returns)
        q = 1 - p
        
        # 2. 賠率 (Odds, b)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0: return 0.5
        b = avg_win / avg_loss
        
        # 3. 凱利公式 (Full Kelly) -> f = p - (q / b)
        f = p - (q / b)
        
        # 4. 安全係數 (Half-Kelly) & 邊界限制
        # 限制單次投入最多不超過現金池的 40% (避免過度激進)
        return max(0.0, min(f * 0.5, 0.4))

    # ==========================================
    # ⏳ 進階回測引擎 (含 Kelly 分配)
    # ==========================================
    def run_advanced_backtest(self, targets, monthly_budget, start_date):
        # 1. 下載數據
        tickers = targets + ['^VIX']
        data = yf.download(tickers, start=start_date, progress=False)
        
        # 兼容性處理：有些版本的 yf 下載後是 MultiIndex
        if 'Close' in data.columns:
            closes = data['Close']
        else:
            closes = data

        # 確保 VIX 存在
        try:
            vix_series = closes['^VIX']
        except:
            vix_series = pd.Series(20, index=closes.index)

        # 2. 準備回測變數
        portfolio_history = []
        
        # Agent 帳戶
        agent_holdings = {t: 0.0 for t in targets}
        agent_cash = 0.0
        
        # DCA (笨定投) 帳戶
        dca_holdings = {t: 0.0 for t in targets}
        
        # 總投入本金計數器
        total_invested = 0.0
        
        # 預熱期 (為了算 MA200)
        valid_dates = closes.index[200:]
        if len(valid_dates) == 0: return None

        # --- 🔧 修復 KeyError 的關鍵段落 ---
        # 建立一個臨時 DataFrame 來找每月第一個交易日
        df_calendar = pd.DataFrame(index=valid_dates)
        df_calendar['YM'] = df_calendar.index.to_period('M')
        
        # Reset index 後，日期欄位名稱可能是 'Date' 或 'index'，我們動態抓取第0欄
        df_reset = df_calendar.reset_index()
        date_col_name = df_reset.columns[0] 
        salary_dates = df_reset.groupby('YM')[date_col_name].first().values
        # -------------------------------------

        alloc_per_asset = monthly_budget / len(targets) # DCA 用平均分配

        # 3. 開始逐日模擬
        for date in valid_dates:
            # --- A. 更新當日淨值 (Mark to Market) ---
            daily_agent_val = agent_cash
            daily_dca_val = 0.0
            current_prices = {}
            
            skip_day = False
            for t in targets:
                try:
                    price = float(closes.loc[date, t])
                    if np.isnan(price) or price <= 0:
                        # 若當日無價(休市)，嘗試抓前值，若無則跳過
                        price = 0 
                    current_prices[t] = price
                    
                    daily_agent_val += agent_holdings[t] * price
                    daily_dca_val += dca_holdings[t] * price
                except:
                    skip_day = True
            
            if skip_day: continue

            # 記錄
            portfolio_history.append({
                "Date": date,
                "Agent策略": daily_agent_val,
                "DCA定投": daily_dca_val,
                "投入本金": total_invested
            })

            # --- B. 發薪日入金 ---
            if date in salary_dates:
                agent_cash += monthly_budget
                total_invested += monthly_budget
                
                # DCA 策略：無腦平均買入
                for t in targets:
                    if current_prices.get(t, 0) > 0:
                        dca_holdings[t] += alloc_per_asset / current_prices[t]

            # --- C. Agent 策略執行 (Kelly Buy) ---
            vix = float(vix_series.loc[date])
            
            # 隨機打散順序，避免每次都先買第一支股票導致現金被吃光
            shuffled_targets = list(targets)
            np.random.shuffle(shuffled_targets)

            for t in shuffled_targets:
                price = current_prices.get(t, 0)
                if price <= 0: continue
                
                # 取得過去數據計算 MA 和 Kelly
                # 使用切片取得直到當天的數據
                past_data = closes[t].loc[:date]
                if len(past_data) < 201: continue
                
                # 為了效能，只取最近 201 筆算均線
                recent_window = past_data.iloc[-201:-1] 
                ma200 = recent_window.mean()
                
                # 訊號判定
                signal_buy = (price > ma200) or (vix > 30)
                
                if signal_buy and agent_cash > 100: # 現金大於 100 才動作
                    # 🔥 重點：在回測中使用 Kelly 公式
                    # 計算過去一年的 Kelly 值
                    kelly_f = self.calculate_kelly_fraction(past_data.iloc[-252:])
                    
                    # 決定投入金額
                    # 這裡設定一個邏輯：如果 Kelly 說買 20%，就是買現金池的 20%
                    invest_amt = agent_cash * kelly_f
                    
                    # 為了防止過度頻繁的小額交易，設定最小交易門檻 (例如 $500)
                    if invest_amt > 500:
                        shares = invest_amt / price
                        agent_holdings[t] += shares
                        agent_cash -= invest_amt

        # 4. 整理結果
        df_history = pd.DataFrame(portfolio_history).set_index("Date")
        
        if df_history.empty:
            return None

        final_agent = df_history['Agent策略'].iloc[-1]
        final_dca = df_history['DCA定投'].iloc[-1]
        years = (df_history.index[-1] - df_history.index[0]).days / 365.25
        
        # 計算年化 (CAGR)
        agent_cagr = (final_agent / total_invested) ** (1/years) - 1 if total_invested > 0 else 0
        dca_cagr = (final_dca / total_invested) ** (1/years) - 1 if total_invested > 0 else 0
        
        return {
            "history_df": df_history,
            "metrics": {
                "years": years,
                "agent_final": final_agent,
                "dca_final": final_dca,
                "agent_cagr": agent_cagr,
                "dca_cagr": dca_cagr,
                "total_invested": total_invested
            }
        }