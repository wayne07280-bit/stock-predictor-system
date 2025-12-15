import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from plotly.subplots import make_subplots # 用於多圖表繪製
from datetime import date, timedelta
import pandas_ta as ta # 技術分析函式庫

# 設定：觀察過去 60 天的數據
TIME_STEP = 60

# --- 1. 輔助函式：數據預處理 ---
def create_dataset(data, time_step=TIME_STEP):
    """將股價序列轉換為適合 LSTM 的 X (特徵) 和 Y (標籤) 數據集"""
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        # 預測的是第 time_step + 1 天的 'Close' 價格 (索引 0)
        Y.append(data[i + time_step, 0]) 
    return np.array(X), np.array(Y)

# --- 2. 輔助函式：建構並訓練 LSTM 模型 ---
def build_and_train_lstm(X_train, y_train, features_count):
    """建立更高複雜度的 3 層 LSTM 模型"""
    model = Sequential()
    
    # 調整：使用 3 層 LSTM，增加模型深度
    model.add(LSTM(128, return_sequences=True, input_shape=(TIME_STEP, features_count))) 
    model.add(Dropout(0.3))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(1)) 
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # EarlyStopping：當驗證損失在 5 個 epochs 內沒有改善時，停止訓練
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # 增加 epochs 到 50
    model.fit(X_train, y_train, validation_split=0.1, batch_size=64, epochs=50, 
              callbacks=[early_stopping], verbose=0)
    
    return model

# --- 3. 核心函式：計算技術指標 ---
def calculate_technical_indicators(df):
    """計算 MACD, RSI, 布林帶 (BBANDS) 和 KD 線 (Stochastics) 等技術指標"""
    
    # 檢查數據是否足夠計算技術指標 (例如: 50日MA 需要 50天數據)
    if len(df) < 60: 
        st.error("❌ 歷史數據不足，無法計算完整的技術指標。請選擇有更多交易記錄的股票。")
        return pd.DataFrame({'Close': []}) 
        
    # 1. 計算 移動平均線 (MA)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True) 
    
    # 2. 計算 RSI
    df.ta.rsi(length=14, append=True) 
    
    # 3. 計算 MACD
    df.ta.macd(append=True)
    
    # 4. 計算 布林帶 (BBANDS)
    df.ta.bbands(length=20, append=True) 
    
    # 5. 計算 KD 線 (Stochastics)
    df.ta.stoch(append=True) 
    
    
    # *** 🛠️ 關鍵修正：使用動態模糊匹配查找 pandas_ta 生成的欄位 🛠️ ***
    
    NEW_COL_MAP = {}
    
    # 遍歷 DataFrame 中所有的欄位
    for col in df.columns:
        col_upper = col.upper()
        
        # --- MA/RSI/MACD 匹配 (大部分版本是固定的) ---
        if col_upper == 'SMA_20':
            NEW_COL_MAP[col] = 'MA_20'
        elif col_upper == 'SMA_50':
            NEW_COL_MAP[col] = 'MA_50'
        elif col_upper == 'RSI_14':
            NEW_COL_MAP[col] = 'RSI'
        elif col_upper.startswith('MACD_12_26'):
            NEW_COL_MAP[col] = 'MACD'
        elif col_upper.startswith('MACDS_12_26'):
            NEW_COL_MAP[col] = 'MACD_Signal'
        
        # --- 布林帶和 KD 線模糊匹配 (確保 BB 欄位存在) ---
        
        # 布林帶 (BBL=Lower, BBM=Middle, BBU=Upper)
        # 匹配任何以 BBL, BBM, BBU 開頭的欄位，忽略後續的週期和標準差數字
        elif col_upper.startswith('BBL'):
            NEW_COL_MAP[col] = 'BB_Lower'
        elif col_upper.startswith('BBU'):
            NEW_COL_MAP[col] = 'BB_Upper'
        elif col_upper.startswith('BBM'):
            NEW_COL_MAP[col] = 'BB_Middle'
            
        # KD 線 (STOCHK=K, STOCHD=D)
        elif col_upper.startswith('STOCHK'):
            NEW_COL_MAP[col] = 'KD_K'
        elif col_upper.startswith('STOCHD'):
            NEW_COL_MAP[col] = 'KD_D'
            
    # 執行重命名
    df.rename(columns=NEW_COL_MAP, inplace=True)
    
    
    # 6. 安全計算 BB_Ratio (布林帶相對位置)
    # 由於我們使用了模糊匹配，BB_Lower 和 BB_Upper 現在應該會存在
    if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
        # 新增一個特徵：收盤價是否接近布林帶上下緣 (正規化至 0-1 區間)
        df['BB_Ratio'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    else:
        # 如果布林帶欄位缺失，則 BB_Ratio 設置為一個常數
        df['BB_Ratio'] = 0.5 

    # 移除 NaN 值 (技術指標計算初期會產生 NaN)
    df.dropna(inplace=True) 
    return df
    
# --- 4. 核心主程式函式 ---
def run_prediction_system(stock_ticker, market_type, predict_days):
    # 設定參數
    TIME_STEP = 60 # 觀察過去 60 天的數據

    st.subheader(f"📊 正在分析股票代號/名稱: **{stock_ticker}**")

    # 處理台股代碼 (預設加上 .TW)
    if market_type == "台股" and not stock_ticker.endswith(('.TW', '.TWO')):
        stock_ticker += ".TW"
        
    # 獲取歷史數據的日期範圍
    start_date = date.today() - timedelta(days=3 * 365)
    end_date = date.today() - timedelta(days=1)
    
    data = pd.DataFrame() # 初始化一個空的 DataFrame
    
    # *** 數據獲取修正：台股雙重查詢嘗試 (.TW / .TWO) ***
    
    try:
        data = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)
    except Exception:
        pass 
    
    # 嘗試 .TWO 後綴
    if data.empty and market_type == "台股":
        base_ticker = stock_ticker.replace('.TW', '').replace('.TWO', '')
        stock_ticker_two = f"{base_ticker}.TWO"
        st.info(f"第一次查詢失敗，嘗試替換為台股後綴: **{stock_ticker_two}**")
        try:
            data = yf.download(stock_ticker_two, start=start_date, end=end_date, progress=False)
            if not data.empty:
                stock_ticker = stock_ticker_two # 更新股票代號
        except Exception:
            pass 

    # *** 數據獲取修正：處理 MultiIndex 欄位名稱問題 ***
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    if data.empty:
        st.warning("⚠️ 查無此股票代號的歷史數據。請確認輸入是否正確。")
        return

    # --- 數據準備與特徵工程 ---
    
    data = calculate_technical_indicators(data.copy())
    
    # 選擇用於訓練模型的特徵 
    all_possible_features = ['Close', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'KD_K', 'KD_D', 'BB_Ratio'] 
    
    features = [f for f in all_possible_features if f in data.columns]
    
    data_for_model = data[features].values
    
    # 2. 數據標準化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_model)
    
    # 3. 建立訓練集
    features_count = len(features)
    X_train, y_train = create_dataset(scaled_data)
    
    if len(X_train) < 100:
        st.error("❌ 歷史數據不足，無法訓練模型。請選擇有更多交易記錄的股票。")
        return
        
    

    # --- 模型訓練 ---
    with st.spinner("掐但幾累喔..."):
        model = build_and_train_lstm(X_train, y_train, features_count) 
    st.success("✅ 模型訓練完成！")
    
    # --- 預測未來 (滾動預測與漲跌幅限制) ---
    last_input = scaled_data[-TIME_STEP:] 
    future_predictions = []
    current_input = last_input
    
    prev_close = data['Close'].iloc[-1] 
    
    for i in range(predict_days):
        prediction = model.predict(current_input.reshape(1, TIME_STEP, features_count), verbose=0)
        
        prediction_scaled = np.zeros((1, features_count)) 
        prediction_scaled[0, 0] = prediction[0, 0]
        real_prediction = scaler.inverse_transform(prediction_scaled)[0, 0]
        
        # *** 🛠️ 修正：台股漲跌幅限制 (+/- 10%) 🛠️ ***
        if market_type == "台股":
            limit_up = prev_close * 1.10
            limit_down = prev_close * 0.90
            constrained_prediction = np.clip(real_prediction, limit_down, limit_up)
            final_prediction = constrained_prediction
        else:
            final_prediction = real_prediction
        
        future_predictions.append(final_prediction)
        
        # 更新輸入數據
        new_feature_values = current_input[-1].copy() 
        temp_scaled = np.zeros((1, features_count)) 
        temp_scaled[0, 0] = final_prediction 
        
        constrained_scaled_close = scaler.transform(temp_scaled)[0, 0] 
        
        new_feature_values[0] = constrained_scaled_close # 更新 'Close' 特徵 (索引 0)
        
        current_input = np.vstack([current_input[1:], new_feature_values])
        
        prev_close = final_prediction 
    
    # --- 繪圖與結果展示 (包含布林通道和 KD 線) ---
    predict_dates = [data.index[-1] + timedelta(days=i) for i in range(1, predict_days + 1)]
    
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.75, 0.25], 
                        subplot_titles=(f'{stock_ticker} 歷史股價、布林通道與預測', 'KD 線 (隨機指標)'))

    # --- 第一行：K 線圖、布林通道和預測線 ---
    
    # 歷史 K 線圖 (Candlestick)
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='歷史K線'
    ), row=1, col=1)

    # 預測線 (Scatter)
    fig.add_trace(go.Scatter(
        x=data.index.tolist()[-TIME_STEP:] + predict_dates, 
        y=data['Close'].tolist()[-TIME_STEP:] + future_predictions,
        mode='lines+markers',
        name=f'預測股價 ({predict_days}天)',
        line=dict(color='orange', width=3)
    ), row=1, col=1)
    
    # *** 🛠️ 關鍵修正：布林通道安全繪圖 🛠️ ***
    bb_upper = data.get('BB_Upper')
    bb_lower = data.get('BB_Lower')
    bb_middle = data.get('BB_Middle') 
    
    if bb_upper is not None and bb_lower is not None:
        # 上軌
        fig.add_trace(go.Scatter(
            x=data.index, y=bb_upper, line=dict(color='gray', width=1, dash='dash'), name='布林帶上軌'
        ), row=1, col=1)
        # 下軌
        fig.add_trace(go.Scatter(
            x=data.index, y=bb_lower, line=dict(color='gray', width=1, dash='dash'), name='布林帶下軌'
        ), row=1, col=1)
        # 中軌
        if bb_middle is not None:
            fig.add_trace(go.Scatter(
                x=data.index, y=bb_middle, line=dict(color='blue', width=1), name='布林帶中軌 (MA20)'
            ), row=1, col=1) 
            
    # --- 第二行：KD 線圖 (Stochastic Oscillator) ---
    if 'KD_K' in data.columns and 'KD_D' in data.columns:
        # K 線
        fig.add_trace(go.Scatter(
            x=data.index, y=data['KD_K'], line=dict(color='red', width=2), name='K 值'
        ), row=2, col=1)
        # D 線
        fig.add_trace(go.Scatter(
            x=data.index, y=data['KD_D'], line=dict(color='green', width=2), name='D 值'
        ), row=2, col=1)
        
        # 繪製超買線 (80) 和超賣線 (20)
        fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1) 

    # --- 佈局設置 ---
    fig.update_layout(height=700, 
                      showlegend=True,
                      xaxis_rangeslider_visible=False) 
    
    fig.update_xaxes(rangeselector_visible=False, row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True) 
    
    # --- 買賣點建議邏輯 ---
    st.markdown("### 🎯 近期最佳買入點與賣出點建議 (結合 LSTM 預測趨勢)")
    
    latest_close = data['Close'].iloc[-1]
    avg_future_price = np.mean(future_predictions)
    
    prediction_change_percent = (avg_future_price - latest_close) / latest_close * 100
    
    buy_advice = []
    sell_advice = []
    latest = data.iloc[-1]
    
    # --- 買入訊號 ---
    if prediction_change_percent >= 1.0 and 'RSI' in latest and latest['RSI'] < 70:
        buy_advice.append(f"📈 **LSTM 強力看漲 (+{prediction_change_percent:.2f}%)**: 預測未來股價有顯著上漲空間。")

    if 'MACD' in latest and 'MACD_Signal' in latest and latest['MACD_Signal'] > 0 and latest['MACD'] > latest['MACD_Signal'] and prediction_change_percent > 0:
        buy_advice.append("💰 **MACD 金叉訊號** (MACD 線上穿訊號線): 動能轉強，結合預測趨勢向上。")
    
    if 'BB_Ratio' in latest and latest['BB_Ratio'] < 0.1 and prediction_change_percent > 0.1: 
        buy_advice.append("📉 **布林帶下軌支撐**: 價格進入布林帶超賣區，預測有反彈機會。")

    # --- 賣出訊號 ---
    if prediction_change_percent <= -1.0 or ('RSI' in latest and latest['RSI'] > 75):
        sell_advice.append(f"📉 **LSTM 強力看跌 ({prediction_change_percent:.2f}%) / RSI 極度超買**: 預測下跌或 RSI 處於極度超買區。")

    if 'MACD' in latest and 'MACD_Signal' in latest and latest['MACD'] < latest['MACD_Signal'] and prediction_change_percent < 0:
        sell_advice.append("🛑 **MACD 死叉訊號**: 短期動能向下突破訊號線，結合預測趨勢向下。")

    if 'BB_Ratio' in latest and latest['BB_Ratio'] > 0.9:
        sell_advice.append("⚠️ **布林帶上軌壓力**: 價格進入布林帶超買區，可能面臨回調壓力。")
    
    # 輸出建議
    if buy_advice:
        st.info("### **🟢 買入建議:**")
        st.markdown('\n'.join([f'* {advice}' for advice in buy_advice]))
    else:
        st.info("🟢 **目前無明確買入訊號**，建議持續觀察。")
        
    if sell_advice:
        st.warning("### **🔴 賣出建議:**")
        st.markdown('\n'.join([f'* {advice}' for advice in sell_advice]))
    else:
        st.warning("🔴 **目前無明確賣出訊號**，建議持有。")

# --- 5. Streamlit 介面佈局 ---
st.set_page_config(page_title="股票預測系統", layout="wide")
st.title("朴寶崴的股票預測系統 (沒洨用版)")
st.markdown("---")

# 側邊欄輸入
st.sidebar.header("🔍 查詢設定")
market = st.sidebar.radio("選擇市場", ("美股", "台股"))
ticker = st.sidebar.text_input("輸入股票代號/名稱 (例如: AAPL 或 2330)", "2330")

# 新增預測天數選擇
predict_days = st.sidebar.select_slider(
    '選擇預測未來天數',
    options=[3, 7, 14, 30],
    value=7
)

if st.sidebar.button("開始預測", type="primary"):
    if ticker:
        run_prediction_system(ticker.upper(), market, predict_days)
    else:
        st.sidebar.error("請輸入股票代號！")
