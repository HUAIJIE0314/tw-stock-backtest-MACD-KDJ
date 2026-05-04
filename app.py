import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. 網頁基本設定 (維持原視覺風格)
# ==========================================
st.set_page_config(page_title="台股日K MJ策略回測", page_icon="📈", layout="wide")
st.title("📈 台股 MJ 指標 (MACD + KDJ) 日K趨勢策略機器人")

# ==========================================
# 1. 資料抓取模組 (保留快取與 API 備援)
# ==========================================
@st.cache_data(ttl=86400)
def get_all_tw_stocks_with_names():
    def extract_codes_and_names(api_url, suffix):
        try:
            res = requests.get(api_url, timeout=10)
            data = res.json()
            if not data: return {}
            code_key = next((k for k in data[0].keys() if k in ['公司代號', '證券代號', '股票代號', 'Code', 'code', 'Symbol']), None)
            name_key = next((k for k in data[0].keys() if k in ['公司簡稱', '證券名稱', '公司名稱', 'Name', 'name', 'CompanyName']), None)
            if not code_key or not name_key: return {}
            return {f"{str(item.get(code_key, '')).strip()}{suffix}": str(item.get(name_key, '')).strip() for item in data if len(str(item.get(code_key, '')).strip()) >= 4}
        except: return {}
    twse = extract_codes_and_names("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW")
    tpex = extract_codes_and_names("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", ".TWO")
    return {**twse, **tpex} if (twse or tpex) else {'2330.TW': '台積電', '2317.TW': '鴻海'}

# ==========================================
# 2. 側邊欄：參數設定
# ==========================================
st.sidebar.header("⚙️ 回測參數設定")
user_ticker = st.sidebar.text_input("股票代號 (如: 2330)", value="2337", max_chars=6)
initial_capital = st.sidebar.number_input("投入本金 (元)", min_value=10000, max_value=10000000, value=500000, step=10000)
backtest_days = st.sidebar.slider("回測天數 (日K)", min_value=30, max_value=3650, value=120, step=10)
resonance_window = st.sidebar.slider("訊號共振窗口 (天)", min_value=1, max_value=10, value=3)

if st.sidebar.button("🚀 執行 MJ 策略回測", use_container_width=True):
    with st.spinner('數據計算中...'):
        stock_dict = get_all_tw_stocks_with_names()
        ticker_full = next((k for k in stock_dict if k.startswith(user_ticker)), f"{user_ticker}.TW")
        stock_name = stock_dict.get(ticker_full, user_ticker)

        # 下載日K資料
        df = yf.download(ticker_full, period=f"{backtest_days}d", interval="1d", progress=False)
        if df.empty:
            st.error("無法取得資料。")
            st.stop()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['RSI'] = df.ta.rsi(length=14)
        stoch = df.ta.stoch(k=9, d=3, smooth_k=3)
        df = pd.concat([df, stoch], axis=1)
        k_col, d_col = stoch.columns[0], stoch.columns[1]
        df['J'] = 3 * df[k_col] - 2 * df[d_col]

        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        m_line, m_hist, m_sig = macd.columns[0], macd.columns[1], macd.columns[2]

        # 策略訊號 (MJ 共振)
        df['J_up_50'] = (df['J'] > 50) & (df['J'].shift(1) <= 50)
        df['MACD_Red'] = (df[m_hist] > 0) & (df[m_hist].shift(1) <= 0)
        df['Trend_OK'] = df[m_line] > 0
        df['J_down_50'] = (df['J'] < 50) & (df['J'].shift(1) >= 50)
        df['MACD_Green'] = (df[m_hist] < 0) & (df[m_hist].shift(1) >= 0)

        # 共振判斷
        df['Buy_Signal'] = (df['Trend_OK']) & (df['J_up_50'].rolling(resonance_window).max() > 0) & (df['MACD_Red'].rolling(resonance_window).max() > 0)
        df['Sell_Signal'] = (df['J_down_50'].rolling(resonance_window).max() > 0) & (df['MACD_Green'].rolling(resonance_window).max() > 0)
        df = df.dropna()

        # 回測循環
        cap, pos, ent_p, ent_d = initial_capital, 0, 0.0, None
        hist, buys, sells, eq_curve = [], [], [], []

        for d, r in df.iterrows():
            px = r['Close']
            if pos == 0 and r['Buy_Signal']:
                shares = int(cap / (px * 1.001425))
                if shares > 0:
                    pos, ent_p, ent_d = shares, px, d
                    cap -= (pos * ent_p * 1.001425)
                    buys.append({'Date': d, 'Price': ent_p})
            elif pos > 0 and r['Sell_Signal']:
                rev = pos * px * (1 - 0.001425 - 0.003)
                profit = rev - (pos * ent_p * 1.001425)
                hist.append({'Buy_Date': ent_d, 'Sell_Date': d, 'Buy_Price': ent_p, 'Sell_Price': px, 'Profit': profit, 'Return(%)': round((profit/(pos*ent_p*1.001425))*100, 2)})
                cap += rev
                sells.append({'Date': d, 'Price': px})
                pos = 0
            eq_curve.append(cap + (pos * px * 0.995))
        df['Equity'] = eq_curve

        # 報告呈現
        st.header(f"📊 {user_ticker} {stock_name} - MJ 策略報告")
        tr = (df['Equity'].iloc[-1] - initial_capital) / initial_capital
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("初始資金", f"{initial_capital:,.0f} 元")
        c2.metric("最終淨值", f"{df['Equity'].iloc[-1]:,.0f} 元", f"{tr * 100:.2f}%")
        c3.metric("交易次數", len(hist))
        c4.metric("勝率", f"{(sum(1 for t in hist if t['Profit'] > 0)/len(hist)*100 if hist else 0):.2f}%")

        if pos > 0:
            with st.expander("⚠️ 檢視目前持有未平倉部位 (截至回測最後一筆)", expanded=True):
                unrealized = (df['Close'].iloc[-1] * pos * 0.995) - (ent_p * pos * 1.001425)
                st.info(f"**買入時間:** {ent_d.strftime('%Y/%m/%d')} | **買入價格:** {ent_p:.2f} 元")
                st.metric("預估未實現損益", f"{unrealized:,.0f} 元", f"{(df['Close'].iloc[-1]-ent_p)/ent_p*100:.2f}%")

        # 視覺化圖表
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15],
                            subplot_titles=("價格與訊號", "RSI", "KDJ", "MACD", "MJ 共振觀測 (MACD柱 + J線)", "總資金曲線"))
        x_str = df.index.strftime('%Y/%m/%d')
        
        # 價格
        fig.add_trace(go.Scatter(x=x_str, y=df['Close'], name='Close', line=dict(color='#d1d5db')), row=1, col=1)
        if buys: fig.add_trace(go.Scatter(x=[b['Date'].strftime('%Y/%m/%d') for b in buys], y=[b['Price'] for b in buys], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=12, color='#22c55e')), row=1, col=1)
        if sells: fig.add_trace(go.Scatter(x=[s['Date'].strftime('%Y/%m/%d') for s in sells], y=[s['Price'] for s in sells], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=12, color='#f97316')), row=1, col=1)

        # 其他指標
        fig.add_trace(go.Scatter(x=x_str, y=df['RSI'], name='RSI', line=dict(color='#8b5cf6')), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df[k_col], name='K', line=dict(color='#f59e0b', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df[d_col], name='D', line=dict(color='#0ea5e9', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df['J'], name='J', line=dict(color='#ec4899', width=1.5)), row=3, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=x_str, y=df[m_line], name='MACD', line=dict(color='#3b82f6')), row=4, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df[m_sig], name='Signal', line=dict(color='#f59e0b')), row=4, col=1)
        fig.add_trace(go.Bar(x=x_str, y=df[m_hist], name='MACD Hist', marker_color=['#ef4444' if v>=0 else '#22c55e' for v in df[m_hist]], opacity=0.5), row=4, col=1)

        # MJ 共振
        df['J_Shifted'] = df['J'] - 50
        fig.add_trace(go.Bar(x=x_str, y=df[m_hist], name='Resonance Hist', marker_color=['#ef4444' if v>=0 else '#22c55e' for v in df[m_hist]], opacity=0.7), row=5, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df['J_Shifted'], name='J-50', line=dict(color='#ec4899', width=2)), row=5, col=1)
        fig.add_hline(y=0, line_color="white", opacity=0.3, row=5, col=1)

        # 資金曲線
        fig.add_trace(go.Scatter(x=x_str, y=df['Equity'], name='Equity', fill='tozeroy', line=dict(color='#10b981')), row=6, col=1)

        # 垂直虛線貫穿
        for b in buys: fig.add_vline(x=b['Date'].strftime('%Y/%m/%d'), line_width=1, line_dash="dash", line_color="#22c55e", opacity=0.4)
        for s in sells: fig.add_vline(x=s['Date'].strftime('%Y/%m/%d'), line_width=1, line_dash="dash", line_color="#f97316", opacity=0.4)

        fig.update_layout(height=1300, hovermode="x unified", template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 交易明細")
        if hist:
            df_t = pd.DataFrame(hist)
            df_t['Buy_Date'] = df_t['Buy_Date'].dt.strftime('%Y/%m/%d')
            df_t['Sell_Date'] = df_t['Sell_Date'].dt.strftime('%Y/%m/%d')
            st.dataframe(df_t, use_container_width=True, hide_index=True)