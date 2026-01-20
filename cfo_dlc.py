# cfo_dlc.py
import pandas as pd
import numpy as np
import yfinance as yf

class AlphaStrategyDLC:
    def __init__(self):
        self.version = "3.1 (Kelly + Advanced Backtest)"

    # ==========================================
    # 🧮 凱利公式核心引擎 (Kelly Criterion)
    # ==========================================
    def calculate_kelly_fraction(self, price_series, window=252):
        """
        計算凱利公式建議的倉位比例 (Half-Kelly)
        f* = (bp - q) / b
        其中 b = 賠率 (平均獲利/平均虧損), p = 勝率, q = 敗率
        """
        if len(price_series) < window: return 0.1 # 數據不足時預設 10%
        
        # 計算日報酬率
        returns = price_series.pct_change().dropna().tail(window)
        
        # 1. 勝率 (Win Rate, p)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) == 0 or len(losses) == 0: return 0.1
        
        p = len(wins) / len(returns)
        q = 1 - p
        
        # 2. 賠率 (Odds, b) -> 平均獲利 / 平均虧損絕對值
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        if avg_loss == 0: return 0.1
        
        b = avg_win / avg_loss
        
        # 3. 凱利公式 (Full Kelly)
        f = p - (q / b)
        
        # 4. 安全係數 (Half-Kelly)
        # 為了避免破產風險，專業機構通常只用凱利值的一半
        safe_f = f * 0.5
        
        # 限制：最少買 0%，最多單次投入現金的 50% (避免過度集中)
        return max(0.0, min(safe_f, 0.5))

    # ==========================================
    # ⏳ 進階回測引擎 (含資產曲線)
    # ==========================================
    def run_advanced_backtest(self, targets, monthly_budget, start_date):
        # 下載數據
        tickers = targets + ['^VIX']
        data = yf.download(tickers, start=start_date, progress=False)
        
        # 格式清洗
        if 'Close' in data.columns: closes = data['Close']
        else: closes = data
            
        # 儲存每天的總資產價值，用於畫圖
        portfolio_history = []
        
        # 初始化回測變數
        agent_holdings = {t: 0 for t in targets} # 各標的股數
        agent_cash = 0
        total_invested = 0
        
        dca_holdings = {t: 0 for t in targets}
        dca_invested = 0
        
        # 確保有 VIX 數據
        try:
            vix_series = closes['^VIX']
        except:
            vix_series = pd.Series(20, index=closes.index) # 假數據防呆

        # 為了計算年線，需要預熱數據，所以從第 200 天開始模擬
        valid_dates = closes.index[200:]
        
        # 判斷每月發薪日
        df_dates = pd.DataFrame(index=valid_dates)
        df_dates['YM'] = df_dates.index.to_period('M')
        salary_dates = df_dates.reset_index().groupby('YM')['index'].first().values
        
        alloc_per_asset = monthly_budget / len(targets)
        
        for date in valid_dates:
            # 1. 計算當日資產總值 (Mark to Market)
            daily_agent_value = agent_cash
            daily_dca_value = 0
            current_prices = {}
            
            skip_day = False
            for t in targets:
                try:
                    price = closes.loc[date, t]
                    if pd.isna(price): 
                        price = 0 # 若當日無數據(休市)，暫用0或前值(這裡簡化)
                    current_prices[t] = price
                    
                    daily_agent_value += agent_holdings[t] * price
                    daily_dca_value += dca_holdings[t] * price
                except:
                    skip_day = True
            
            if skip_day: continue

            # 記錄這一天的淨值
            portfolio_history.append({
                "Date": date,
                "Agent_Equity": daily_agent_value,
                "DCA_Equity": daily_dca_value,
                "Invested_Capital": total_invested
            })
            
            # 2. 發薪日入金
            if date in salary_dates:
                agent_cash += monthly_budget
                total_invested += monthly_budget
                dca_invested += monthly_budget
                
                # DCA 策略：無腦買入
                for t in targets:
                    if current_prices[t] > 0:
                        dca_holdings[t] += alloc_per_asset / current_prices[t]

            # 3. Agent 策略執行 (每日盤後檢查)
            vix = vix_series.loc[date]
            
            for t in targets:
                if current_prices[t] <= 0: continue
                
                # 取得該標的過去 200 天價格 (切片)
                # 注意：這裡效率較低，但為了準確模擬動態年線
                past_prices = closes[t].loc[:date].tail(201)[:-1] 
                if len(past_prices) < 200: continue
                
                ma200 = past_prices.mean()
                price = current_prices[t]
                
                # 訊號判定
                signal_buy = (price > ma200) or (vix > 30)
                
                # 執行買入：使用 Kelly 公式決定買多少
                if signal_buy and agent_cash > 0:
                    # 針對該標的計算 Kelly 係數
                    # (使用過去 1 年數據算勝率)
                    kelly_f = self.calculate_kelly_fraction(past_prices, window=252)
                    
                    # 決定投入金額： 現金 * Kelly比例
                    # 為了避免過於激進，我們再除以標的數量，避免單一標的吃光現金
                    invest_amt = agent_cash * kelly_f
                    
                    # 買入
                    if invest_amt > 0:
                        shares_to_buy = invest_amt / price
                        agent_holdings[t] += shares_to_buy
                        agent_cash -= invest_amt

        # 整理結果
        df_history = pd.DataFrame(portfolio_history).set_index("Date")
        
        # 計算最終指標
        final_agent = df_history['Agent_Equity'].iloc[-1]
        final_dca = df_history['DCA_Equity'].iloc[-1]
        years = (df_history.index[-1] - df_history.index[0]).days / 365.25
        
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