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
st.title("📈 台股 MJ 指標 (MACD + KDJ) 日K趨勢策略機器人")

# ==========================================
# 1. 資料抓取模組 (保留原本強大的搜尋與搜尋備援)
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
            return {f"{str(item.get(code_key, '')).strip()}{suffix}": str(item.get(name_key, '')).strip() for item in data}
        except: return {}

    twse = extract_codes_and_names("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW")
    tpex = extract_codes_and_names("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", ".TWO")
    return {**twse, **tpex} if (twse or tpex) else {'2330.TW': '台積電', '2317.TW': '鴻海'}

# ==========================================
# 2. 側邊欄：參數設定
# ==========================================
st.sidebar.header("⚙️ 回測參數設定")
user_ticker = st.sidebar.text_input("股票代號 (如: 6510, 2330)", value="6510", max_chars=6)
initial_capital = st.sidebar.number_input("投入本金 (元)", min_value=10000, value=500000, step=10000)
backtest_days = st.sidebar.slider("回測天數 (日K)", min_value=30, max_value=3650, value=730, step=30)
resonance_window = st.sidebar.slider("訊號共振窗口 (天)", min_value=1, max_value=10, value=3)

if st.sidebar.button("🚀 執行 MJ 策略回測", use_container_width=True):
    with st.spinner('正在精確分析 MJ 指標數據...'):
        stock_dict = get_all_tw_stocks_with_names()
        filtered_list = {k: v for k, v in stock_dict.items() if k.startswith(user_ticker)}
        
        if filtered_list:
            ticker_full = list(filtered_list.keys())[0]
            stock_name = filtered_list[ticker_full]
        else:
            ticker_candidates = [f"{user_ticker}.TWO", f"{user_ticker}.TW", user_ticker]
            ticker_full = None
            stock_name = user_ticker
            for candidate in ticker_candidates:
                try:
                    test_df = yf.download(candidate, period="1d", progress=False)
                    if not test_df.empty:
                        ticker_full = candidate
                        break
                except: pass
            
            if not ticker_full:
                st.error(f"找不到代號：{user_ticker}")
                st.stop()

        df = yf.download(ticker_full, period=f"{backtest_days}d", interval="1d", progress=False)
        if df.empty:
            st.error("無法取得歷史資料。")
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

        # MJ 策略核心
        df['J_up_50'] = (df['J'] > 50) & (df['J'].shift(1) <= 50)
        df['MACD_Red'] = (df[m_hist] > 0) & (df[m_hist].shift(1) <= 0)
        df['Trend_OK'] = df[m_line] > 0
        df['J_down_50'] = (df['J'] < 50) & (df['J'].shift(1) >= 50)
        df['MACD_Green'] = (df[m_hist] < 0) & (df[m_hist].shift(1) >= 0)

        df['Buy_Signal'] = (df['Trend_OK']) & (df['J_up_50'].rolling(resonance_window).max() > 0) & (df['MACD_Red'].rolling(resonance_window).max() > 0)
        df['Sell_Signal'] = (df['J_down_50'].rolling(resonance_window).max() > 0) & (df['MACD_Green'].rolling(resonance_window).max() > 0)
        df = df.dropna()

        # 回測邏輯
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
                hist.append({'Buy_Date': ent_d, 'Sell_Date': d, 'Profit': profit, 'Return(%)': round((profit/(pos*ent_p*1.001425))*100, 2)})
                cap += rev
                sells.append({'Date': d, 'Price': px})
                pos = 0
            eq_curve.append(cap + (pos * px * 0.995))
        df['Equity'] = eq_curve

        # 報告呈現
        st.header(f"📊 {ticker_full} {stock_name} - MJ 策略報告")
        if pos > 0:
            with st.expander("⚠️ 檢視目前持有未平倉部位 (截至回測最後一筆)", expanded=True):
                unrealized = (df['Close'].iloc[-1] * pos * 0.995) - (ent_p * pos * 1.001425)
                st.info(f"**買入時間:** {ent_d.strftime('%Y/%m/%d')} | **買入價格:** {ent_p:.2f} 元")
                st.metric("預估未實現損益", f"{unrealized:,.0f} 元", f"{(df['Close'].iloc[-1]-ent_p)/ent_p*100:.2f}%")

        # 視覺化圖表
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                            row_heights=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15],
                            subplot_titles=("價格與訊號", "RSI (14)", "KDJ (9,3,3)", "MACD (12,26,9)", "MJ 共振觀測 (MACD柱 + J線對齊)", "總資金曲線"),
                            specs=[[{"secondary_y": False}],[{"secondary_y": False}],[{"secondary_y": False}],[{"secondary_y": False}],[{"secondary_y": True}],[{"secondary_y": False}]])
        
        x_str = df.index.strftime('%Y/%m/%d')
        
        # 1. 價格
        fig.add_trace(go.Scatter(x=x_str, y=df['Close'], name='收盤價', line=dict(color='#d1d5db')), row=1, col=1)
        if buys: fig.add_trace(go.Scatter(x=[b['Date'].strftime('%Y/%m/%d') for b in buys], y=[b['Price'] for b in buys], mode='markers', name='買入點', marker=dict(symbol='triangle-up', size=12, color='#22c55e')), row=1, col=1)
        if sells: fig.add_trace(go.Scatter(x=[s['Date'].strftime('%Y/%m/%d') for s in sells], y=[s['Price'] for s in sells], mode='markers', name='賣出點', marker=dict(symbol='triangle-down', size=12, color='#f97316')), row=1, col=1)

        # 2. RSI / 3. KDJ
        fig.add_trace(go.Scatter(x=x_str, y=df['RSI'], name='RSI', line=dict(color='#8b5cf6')), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df[k_col], name='K線', line=dict(color='#f59e0b', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df[d_col], name='D線', line=dict(color='#0ea5e9', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df['J'], name='J線', line=dict(color='#ec4899', width=1.5)), row=3, col=1)
        
        # 4. MACD
        fig.add_trace(go.Scatter(x=x_str, y=df[m_line], name='MACD快線', line=dict(color='#3b82f6')), row=4, col=1)
        fig.add_trace(go.Scatter(x=x_str, y=df[m_sig], name='Signal慢線', line=dict(color='#f59e0b')), row=4, col=1)
        fig.add_trace(go.Bar(x=x_str, y=df[m_hist], name='MACD柱狀體', marker_color=['#ef4444' if v>=0 else '#22c55e' for v in df[m_hist]], opacity=0.5), row=4, col=1)

        # 5. MJ 共振 (🌟 0 軸水平對齊)
        df['J_Shifted'] = df['J'] - 50
        fig.add_trace(go.Bar(x=x_str, y=df[m_hist], name='共振柱狀圖', marker_color=['#ef4444' if v>=0 else '#22c55e' for v in df[m_hist]], opacity=0.7), row=5, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x_str, y=df['J_Shifted'], name='J線 (對齊0軸)', line=dict(color='#ec4899', width=2)), row=5, col=1, secondary_y=True)
        
        max_m = df[m_hist].abs().max() * 1.1
        max_j = df['J_Shifted'].abs().max() * 1.1
        fig.update_yaxes(range=[-max_m, max_m], row=5, col=1, secondary_y=False)
        fig.update_yaxes(range=[-max_j, max_j], row=5, col=1, secondary_y=True)
        fig.add_hline(y=0, line_color="white", opacity=0.3, row=5, col=1)

        # 6. 總資金
        fig.add_trace(go.Scatter(x=x_str, y=df['Equity'], name='帳戶價值', fill='tozeroy', line=dict(color='#10b981')), row=6, col=1)

        # 垂直買賣線貫穿
        for b in buys: fig.add_vline(x=b['Date'].strftime('%Y/%m/%d'), line_width=1, line_dash="dash", line_color="#22c55e", opacity=0.4)
        for s in sells: fig.add_vline(x=s['Date'].strftime('%Y/%m/%d'), line_width=1, line_dash="dash", line_color="#f97316", opacity=0.4)

        # 🌟 設定圖例 (Legend) 在上方水平排列
        fig.update_layout(
            height=1300, 
            hovermode="x unified", 
            template="plotly_dark", 
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 交易明細")
        if hist:
            df_t = pd.DataFrame(hist)
            df_t['Buy_Date'] = df_t['Buy_Date'].dt.strftime('%Y/%m/%d')
            df_t['Sell_Date'] = df_t['Sell_Date'].dt.strftime('%Y/%m/%d')
            st.dataframe(df_t, use_container_width=True, hide_index=True)