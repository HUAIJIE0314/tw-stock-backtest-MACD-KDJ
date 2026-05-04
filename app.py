import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. 網頁基本設定
# ==========================================
st.set_page_config(page_title="台股日K MJ策略回測", page_icon="📈", layout="wide")
st.title("📈 台股 MJ 指標 (MACD + KDJ) 日K共振策略機器人")

# ==========================================
# 1. 資料抓取模組 (使用快取)
# ==========================================
@st.cache_data(ttl=86400)
def get_all_tw_stocks_with_names():
    """抓取台股與櫃買中心股票代號字典"""
    def extract_codes_and_names(api_url, suffix):
        try:
            res = requests.get(api_url, timeout=10)
            data = res.json()
            if not data: return {}
            code_key = next((k for k in data[0].keys() if k in ['公司代號', '證券代號', '股票代號']), 'Symbol')
            name_key = next((k for k in data[0].keys() if k in ['公司簡稱', '證券名稱', '公司名稱']), 'Name')
            stock_dict = {}
            for item in data:
                code = str(item.get(code_key, '')).strip()
                name = str(item.get(name_key, '')).strip()
                if code and len(code) >= 4: 
                    stock_dict[f"{code}{suffix}"] = name
            return stock_dict
        except Exception: return {}

    twse = extract_codes_and_names("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW")
    tpex = extract_codes_and_names("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", ".TWO")
    return {**twse, **tpex} if (twse or tpex) else {'2330.TW': '台積電', '2317.TW': '鴻海'}

# ==========================================
# 2. 側邊欄：參數設定
# ==========================================
st.sidebar.header("⚙️ MJ 策略參數設定")
user_ticker = st.sidebar.text_input("股票代號 (如: 2330, 0050)", value="2330", max_chars=6)
initial_capital = st.sidebar.number_input("投入本金 (元)", min_value=10000, value=500000, step=10000)
backtest_years = st.sidebar.slider("回測長度 (年)", min_value=1, max_value=10, value=3)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 策略容錯設定")
lookback_window = st.sidebar.slider("訊號共振容許天數 (建議 3-5 天)", 1, 10, 3)
st.sidebar.info("💡 當 MACD 翻紅與 J 線破 50 不同步時，在指定天數內皆發生視為訊號。")

if st.sidebar.button("🚀 執行 MJ 策略回測", use_container_width=True):
    with st.spinner('正在分析日K數據中...'):
        stock_dict = get_all_tw_stocks_with_names()
        # 尋找完整代號
        ticker_full = next((k for k in stock_dict if k.startswith(user_ticker)), None)
        if not ticker_full:
            ticker_full = f"{user_ticker}.TW" # 預設補點 TW
        stock_name = stock_dict.get(ticker_full, user_ticker)

        # 1. 下載日K資料
        df = yf.download(ticker_full, period=f"{backtest_years}y", interval="1d", progress=False)
        if df.empty:
            st.error("無法取得歷史資料。")
            st.stop()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. 計算技術指標 (KDJ, MACD)
        # MACD (12, 26, 9)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        m_line = macd.columns[0] # MACD_12_26_9
        m_hist = macd.columns[1] # MACDh_12_26_9

        # KDJ (9, 3, 3) - 手動計算 J 線
        stoch = df.ta.stoch(k=9, d=3, smooth_k=3)
        df = pd.concat([df, stoch], axis=1)
        k_col = stoch.columns[0]
        d_col = stoch.columns[1]
        df['J'] = 3 * df[k_col] - 2 * df[d_col]

        # 3. 產生進出場訊號
        # 基礎條件 (不含未來資料)
        df['J_up_50'] = (df['J'] > 50) & (df['J'].shift(1) <= 50)
        df['J_down_50'] = (df['J'] < 50) & (df['J'].shift(1) >= 50)
        df['MACD_to_Red'] = (df[m_hist] > 0) & (df[m_hist].shift(1) <= 0)
        df['MACD_to_Green'] = (df[m_hist] < 0) & (df[m_hist].shift(1) >= 0)
        
        # 趨勢過濾: MACD 快線必須在零軸之上
        df['Trend_Filter'] = df[m_line] > 0

        # 共振邏輯 (放寬同步性，使用 rolling window 檢查近 N 天是否發生過)
        # 買入: 趨勢向上 & 近 N 天內出現過 J 破 50 & 近 N 天內出現過 MACD 翻紅
        df['Buy_Signal'] = (df['Trend_Filter']) & \
                           (df['J_up_50'].rolling(window=lookback_window).max() > 0) & \
                           (df['MACD_to_Red'].rolling(window=lookback_window).max() > 0)

        # 賣出: 近 N 天內出現過 J 跌破 50 & 近 N 天內出現過 MACD 翻綠
        df['Sell_Signal'] = (df['J_down_50'].rolling(window=lookback_window).max() > 0) & \
                            (df['MACD_to_Green'].rolling(window=lookback_window).max() > 0)

        df = df.dropna()

        # 4. 回測邏輯
        capital = initial_capital
        position = 0
        entry_price = 0.0
        trade_history = []
        equity_curve = []
        buy_points = []
        sell_points = []

        for date, row in df.iterrows():
            current_price = row['Close']
            if position == 0 and row['Buy_Signal']:
                shares = int(capital / (current_price * 1.001425))
                if shares > 0:
                    position = shares
                    entry_price = current_price
                    capital -= (position * entry_price * 1.001425)
                    buy_points.append({'Date': date, 'Price': entry_price})
            elif position > 0 and row['Sell_Signal']:
                revenue = position * current_price * (1 - 0.001425 - 0.003)
                profit = revenue - (position * entry_price * 1.001425)
                trade_history.append({
                    'Buy_Date': buy_points[-1]['Date'], 'Sell_Date': date,
                    'Profit': profit, 'Return(%)': (profit / (position * entry_price * 1.001425)) * 100
                })
                capital += revenue
                sell_points.append({'Date': date, 'Price': current_price})
                position = 0

            current_val = capital + (position * current_price * 0.995)
            equity_curve.append(current_val)
        
        df['Equity'] = equity_curve

        # 5. 視覺化
        st.header(f"📊 {ticker_full} {stock_name} MJ 策略報告")
        
        # 指標卡
        total_ret = (df['Equity'].iloc[-1] - initial_capital) / initial_capital
        c1, c2, c3 = st.columns(3)
        c1.metric("最終淨值", f"{df['Equity'].iloc[-1]:,.0f}", f"{total_ret*100:.2f}%")
        c2.metric("交易次數", len(trade_history))
        c3.metric("目前狀態", "持股中" if position > 0 else "空手")

        # 圖表
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.4, 0.15, 0.25, 0.2],
            subplot_titles=("股價與訊號", "KDJ 指標", "MACD 柱狀體 + J 線 (共振觀測)", "資金曲線")
        )

        x_axis = df.index.strftime('%Y-%m-%d')
        
        # Row 1: Price
        fig.add_trace(go.Scatter(x=x_axis, y=df['Close'], name='收盤價', line=dict(color='gray')), row=1, col=1)
        if buy_points:
            fig.add_trace(go.Scatter(x=[b['Date'].strftime('%Y-%m-%d') for b in buy_points], 
                                     y=[b['Price'] for b in buy_points], mode='markers', name='買入',
                                     marker=dict(symbol='triangle-up', size=12, color='green')), row=1, col=1)
        if sell_points:
            fig.add_trace(go.Scatter(x=[s['Date'].strftime('%Y-%m-%d') for s in sell_points], 
                                     y=[s['Price'] for s in sell_points], mode='markers', name='賣出',
                                     marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)

        # Row 2: KDJ
        fig.add_trace(go.Scatter(x=x_axis, y=df['J'], name='J線', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)

        # Row 3: MACD Hist + J 共振 (關鍵新功能)
        # 將 J 線平移 (J-50) 使其 0 軸與 MACD 對齊 [cite: 17, 22]
        df['J_Norm'] = df['J'] - 50 
        colors = ['green' if v > 0 else 'red' for v in df[m_hist]]
        fig.add_trace(go.Bar(x=x_axis, y=df[m_hist], name='MACD柱狀體', marker_color=colors, opacity=0.5), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=df['J_Norm'], name='J線 (對齊0軸)', line=dict(color='yellow', width=2)), row=3, col=1)
        fig.add_hline(y=0, line_color="white", row=3, col=1)

        # Row 4: Equity
        fig.add_trace(go.Scatter(x=x_axis, y=df['Equity'], name='帳戶價值', fill='tozeroy'), row=4, col=1)

        fig.update_layout(height=900, template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 交易明細
        if trade_history:
            st.subheader("📋 交易明細")
            st.table(pd.DataFrame(trade_history))