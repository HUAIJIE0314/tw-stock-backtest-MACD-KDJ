import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. 網頁基本設定 (保留原介面一致性)
# ==========================================
st.set_page_config(page_title="台股 MJ 指標趨勢策略回測", page_icon="📈", layout="wide")
st.title("📈 台股 MJ 指標 (MACD + KDJ) 策略回測機器人")

# ==========================================
# 1. 資料抓取模組 (保留快取機制)
# ==========================================
@st.cache_data(ttl=86400)
def get_all_tw_stocks_with_names():
    """使用政府 Open API 抓取最新股票代號與【中文簡稱】的字典"""
    def extract_codes_and_names(api_url, suffix):
        try:
            res = requests.get(api_url, timeout=10)
            data = res.json()
            if not data: return {}
            code_key = next((k for k in data[0].keys() if k in ['公司代號', '證券代號', '股票代號', 'Code', 'code', 'Symbol']), None)
            name_key = next((k for k in data[0].keys() if k in ['公司簡稱', '證券名稱', '公司名稱', 'Name', 'name', 'CompanyName']), None)
            if not code_key or not name_key: return {}
            stock_dict = {}
            for item in data:
                code = str(item.get(code_key, '')).strip()
                name = str(item.get(name_key, '')).strip()
                if code and len(code) >= 4: 
                    stock_dict[f"{code}{suffix}"] = name
            return stock_dict
        except Exception: return {}

    twse_dict = extract_codes_and_names("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW")
    tpex_dict = extract_codes_and_names("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", ".TWO")
    full_stock_dict = {**twse_dict, **tpex_dict}
    return full_stock_dict if full_stock_dict else {'2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科'}

# ==========================================
# 2. 側邊欄：參數設定 (保留原始控制項)
# ==========================================
st.sidebar.header("⚙️ 回測參數設定")
user_ticker = st.sidebar.text_input("股票代號 (支持4-6碼)", value="2337", max_chars=6)
initial_capital = st.sidebar.number_input("投入本金 (元)", min_value=10000, max_value=10000000, value=500000, step=10000)
backtest_days = st.sidebar.slider("回測天數 (日K模式)", min_value=30, max_value=3650, value=730, step=30)
resonance_window = st.sidebar.slider("共振容許天數 (訊號放寬)", min_value=1, max_value=10, value=3)

if st.sidebar.button("🚀 執行 MJ 策略回測", use_container_width=True):
    with st.spinner('正在分析數據中...'):
        stock_dict = get_all_tw_stocks_with_names()
        filtered_list = {k: v for k, v in stock_dict.items() if k.startswith(user_ticker)}
        
        if filtered_list:
            ticker_full = list(filtered_list.keys())[0]
            stock_name = filtered_list[ticker_full]
        else:
            ticker_full = f"{user_ticker}.TW"
            stock_name = user_ticker

        # 下載資料 (改為日K)
        df = yf.download(ticker_full, period=f"{backtest_days}d", interval="1d", progress=False)
        if df.empty:
            st.error("無法取得歷史資料。")
            st.stop()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 計算技術指標
        df['RSI'] = df.ta.rsi(length=14)
        # KDJ 計算 (9, 3, 3) [cite: 8]
        stoch = df.ta.stoch(k=9, d=3, smooth_k=3)
        df = pd.concat([df, stoch], axis=1)
        k_col, d_col = stoch.columns[0], stoch.columns[1]
        df['J'] = 3 * df[k_col] - 2 * df[d_col] # J線公式 [cite: 8, 14]

        # MACD 計算
        macd_df = df.ta.macd(fast=12, slow=26, signal=9)
        df = pd.concat([df, macd_df], axis=1)
        m_line, m_hist, m_sig = macd_df.columns[0], macd_df.columns[1], macd_df.columns[2]

        # ==========================================
        # 🌟 MJ 策略邏輯 (加入寬容度與趨勢過濾)
        # ==========================================
        # 買入基礎條件
        df['J_up_50'] = (df['J'] > 50) & (df['J'].shift(1) <= 50) # J線向上突破50(對應MACD 0軸) [cite: 17, 18]
        df['MACD_Red'] = (df[m_hist] > 0) & (df[m_hist].shift(1) <= 0) # MACD柱狀體翻紅 [cite: 16]
        df['Trend_Filter'] = df[m_line] > 0 # 大趨勢過濾：MACD快線在零軸之上 [cite: 17]

        # 賣出基礎條件
        df['J_down_50'] = (df['J'] < 50) & (df['J'].shift(1) >= 50)
        df['MACD_Green'] = (df[m_hist] < 0) & (df[m_hist].shift(1) >= 0)

        # 共振判斷 (使用 rolling 確認窗口內是否兩者都發生) [cite: 22]
        df['Buy_Signal'] = (df['Trend_Filter']) & \
                           (df['J_up_50'].rolling(resonance_window).max() > 0) & \
                           (df['MACD_Red'].rolling(resonance_window).max() > 0)
        
        df['Sell_Signal'] = (df['J_down_50'].rolling(resonance_window).max() > 0) & \
                            (df['MACD_Green'].rolling(resonance_window).max() > 0)

        df = df.dropna()

        # 執行回測邏輯 (事件驅動)
        capital, position, entry_price, entry_date = initial_capital, 0, 0.0, None
        trade_history, buy_points, sell_points, equity_curve = [], [], [], []

        for date, row in df.iterrows():
            curr_p = row['Close']
            if position == 0 and row['Buy_Signal']:
                shares = int(capital / (curr_p * 1.001425))
                if shares > 0:
                    position, entry_price, entry_date = shares, curr_p, date
                    capital -= (position * entry_price * 1.001425)
                    buy_points.append({'Date': date, 'Price': entry_price})
            elif position > 0 and row['Sell_Signal']:
                rev = position * curr_p * (1 - 0.001425 - 0.003)
                profit = rev - (position * entry_price * 1.001425)
                trade_history.append({
                    'Buy_Date': entry_date, 'Sell_Date': date, 'Buy_Price': entry_price,
                    'Sell_Price': curr_p, 'Profit': profit, 'Return(%)': round((profit/(position*entry_price*1.001425))*100, 2)
                })
                capital += rev
                sell_points.append({'Date': date, 'Price': curr_p})
                position = 0

            equity_curve.append(capital + (position * curr_p * 0.995))
        
        df['Equity_Curve'] = equity_curve

        # ==========================================
        # 3. 畫面呈現：回測報告
        # ==========================================
        st.header(f"📊 {user_ticker} {stock_name} - MJ 策略回測報告")
        total_ret = (df['Equity_Curve'].iloc[-1] - initial_capital) / initial_capital
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("初始資金", f"{initial_capital:,.0f} 元")
        c2.metric("最終淨值", f"{df['Equity_Curve'].iloc[-1]:,.0f} 元", f"{total_ret * 100:.2f}%")
        c3.metric("總交易次數", len(trade_history))
        c4.metric("勝率", f"{(sum(1 for t in trade_history if t['Profit'] > 0)/len(trade_history)*100 if trade_history else 0):.2f}%")

        if position > 0:
            unrealized = (df['Close'].iloc[-1] * position * 0.995) - (entry_price * position * 1.001425)
            with st.expander("⚠️ 檢視目前持有未平倉部位 (截至回測最後一筆)", expanded=True):
                st.info(f"**買入時間:** {entry_date.strftime('%Y/%m/%d')} | **買入價格:** {entry_price:.2f} 元")
                st.metric("預估未實現損益", f"{unrealized:,.0f} 元", f"{(df['Close'].iloc[-1]-entry_price)/entry_price*100:.2f}%")

        # ==========================================
        # 4. 畫面呈現：視覺化圖表 (6 個子圖)
        # ==========================================
        fig = make_subplots(
            rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15],
            subplot_titles=("股價與買賣訊號", "RSI (14)", "KDJ 指標", "MACD 指標", "MJ 共振觀測 (MACD柱 + J線)", "總資金曲線")
        )

        x_str = df.index.strftime('%Y/%m/%d') # 修復收盤時間顯示問題
        
        # 價格與訊號
        fig.add_trace(go.Scatter(x=x_str, y=df['Close'], name='Close', line=dict(color='#d1d5db')), row=1, col=1)
        if buy_points:
            fig.add_trace(go.Scatter(x=[b['Date'].strftime('%Y/%m/%d') for b in buy_points], y=[b['Price'] for b in buy_points], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=12, color='#22c55e')), row=1, col=1)
        if sell_points:
            fig.add_trace(go.Scatter(x=[s['Date'].strftime('%Y/%m/%d') for s in sell_points], y=[s['Price'] for s in sell_points], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=12, color='#f97316')), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=x_str, y=df['RSI'], name='RSI', line=dict(color='#8b5cf6')), row=2, col=1)

        # KDJ
        fig.add_trace(go.Scatter(x=x_str, y=df[k_col], name='K', line=dict(color='#f59e0b', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df[d_col], name='D', line=dict(color='#0ea5e9', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df['J'], name='J', line=dict(color='#ec4899', width=1.5)), row=3, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=x_str, y=df[m_line], name='MACD', line=dict(color='#3b82f6')), row=4, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df[m_sig], name='Signal', line=dict(color='#f59e0b')), row=4, col=1)

        # MJ 共振觀測 (MACD柱 + J線疊加) [cite: 22]
        # J 線平移 50 使其與 MACD 0軸對齊
        df['J_Shifted'] = df['J'] - 50
        m_colors = ['#ef4444' if v >= 0 else '#22c55e' for v in df[m_hist]]
        fig.add_trace(go.Bar(x=x_str, y=df[m_hist], name='MACD Hist', marker_color=m_colors, opacity=0.5), row=5, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df['J_Shifted'], name='J-50 (對齊0軸)', line=dict(color='#ec4899', width=2)), row=5, col=1)
        fig.add_hline(y=0, line_color="white", opacity=0.3, row=5, col=1)

        # 資金曲線
        fig.add_trace(go.Scatter(x=x_str, y=df['Equity_Curve'], name='Equity', fill='tozeroy', line=dict(color='#10b981')), row=6, col=1)

        fig.update_layout(height=1200, hovermode="x unified", template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # 5. 交易明細 (保留)
        # ==========================================
        st.subheader("📋 交易明細")
        if trade_history:
            df_t = pd.DataFrame(trade_history)
            df_t['Buy_Date'] = df_t['Buy_Date'].dt.strftime('%Y/%m/%d')
            df_t['Sell_Date'] = df_t['Sell_Date'].dt.strftime('%Y/%m/%d')
            st.dataframe(df_t, use_container_width=True, hide_index=True)
        else:
            st.write("此期間無交易紀錄。")