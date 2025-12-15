import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from datetime import date, timedelta
import pandas_ta as ta # 使用 pandas_ta 替代 talib，提高部署成功率

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

# --- 3. 核心函式：計算技術指標 (使用 pandas_ta) ---
def calculate_technical_indicators(df):
    """計算 MACD, RSI 和 布林帶 (BBANDS) 等技術指標"""
    
    # 檢查數據是否足夠計算技術指標
    if len(df) < 60: # 確保至少有 60 天數據來計算 50日MA和60天時間步
        st.error("❌ 歷史數據不足，無法計算完整的技術指標。請選擇有更多交易記錄的股票。")
        return pd.DataFrame({'Close': []}) # 返回空 DataFrame
        
    # 1. 計算 移動平均線 (MA)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True) 
    
    # 2. 計算 RSI
    df.ta.rsi(length=14, append=True) 
    
    # 3. 計算 MACD
    df.ta.macd(append=True)
    
    # 4. 計算 布林帶 (BBANDS)
    # 確保欄位名稱使用預設，並直接添加到 df 中 (append=True)
    df.ta.bbands(length=20, append=True) 
    
    
    # *** 🛠️ 關鍵修改：使用預設名稱，並進行安全重命名 🛠️ ***
    # pandas_ta 預設生成的欄位名稱 (以週期20，標準差2.0為例)
    # 我們將所有用到的欄位都納入重命名，即使它可能已經是我們想要的名稱
    rename_dict = {
        'SMA_20': 'MA_20', 
        'SMA_50': 'MA_50',
        'RSI_14': 'RSI',
        'MACD_12_26_9': 'MACD',
        'MACDs_12_26_9': 'MACD_Signal',
        'BBL_20_2.0': 'BB_Lower',  # 布林下軌
        'BBU_20_2.0': 'BB_Upper',  # 布林上軌
        'BBM_20_2.0': 'BB_Middle', # 布林中軌
    }

    # 只重命名 DataFrame 中存在的欄位
    final_rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    df.rename(columns=final_rename_dict, inplace=True)
    
    
    # 5. 安全計算 BB_Ratio
    # 確保 BB_Lower 和 BB_Upper 存在，才計算 BB_Ratio，否則賦予預設值
    if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
        # 新增一個特徵：收盤價是否接近布林帶上下緣 (正規化至 0-1 區間)
        df['BB_Ratio'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    else:
        # 如果布林帶欄位缺失，則 BB_Ratio 設置為一個常數，確保模型輸入維度一致性
        df['BB_Ratio'] = 0.5 

    # 移除 NaN 值 (技術指標計算初期會產生 NaN)
    df.dropna(inplace=True) 
    return df

# --- 4. 核心主程式邏輯 ---
def run_prediction_system(stock_ticker, market_type, predict_days):
    # 設定參數
    TIME_STEP = 60 # 觀察過去 60 天的數據

    st.subheader(f"📊 正在分析股票代號/名稱: **{stock_ticker}**")

    # 處理台股代碼
    if market_type == "台股" and not stock_ticker.endswith(('.TW', '.TWO')):
        stock_ticker += ".TW"
        
    try:
        start_date = date.today() - timedelta(days=3 * 365)
        end_date = date.today() - timedelta(days=1)
        # yfinance 獲取數據
        data = yf.download(stock_ticker, start=start_date, end=end_date)
        
        # *** 🛠️ 關鍵修改 (1): 處理 yfinance 可能返回的 MultiIndex 欄位名稱問題 🛠️ ***
        if isinstance(data.columns, pd.MultiIndex):
            # 如果是多重索引，則將其扁平化
            data.columns = [col[0] for col in data.columns]
        
    except Exception as e:
        st.error(f"⚠️ 獲取數據時發生錯誤。請檢查股票代號是否正確。錯誤訊息: {e}")
        return

    if data.empty:
        st.warning("⚠️ 查無此股票代號的歷史數據。請確認輸入是否正確。")
        return

    # --- 數據準備 ---
    # 1. 計算優化後的技術指標 (此函式已包含命名修正和錯誤檢查)
    data = calculate_technical_indicators(data.copy())
    
    # 選擇用於訓練模型的特徵 (收盤價 + 所有的技術指標)
    # 這裡的列表應包含所有可能的特徵名稱
    all_possible_features = ['Close', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Ratio'] 
    
    # 篩選出 data 中實際存在的欄位作為最終特徵
    features = [f for f in all_possible_features if f in data.columns]
    
    st.info(f"💡 本次訓練使用的特徵：{', '.join(features)}")
    
    # 使用篩選後的 features 列表
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
    with st.spinner("🤖 正在訓練 LSTM 模型... (這次訓練更久但更精準！)"):
        # 傳遞 features_count 給建構函式
        model = build_and_train_lstm(X_train, y_train, features_count) 
    st.success("✅ 模型訓練完成！")
    
    # --- 預測未來 (滾動預測) ---
    last_input = scaled_data[-TIME_STEP:] 
    future_predictions = []
    current_input = last_input
    
    for i in range(predict_days):
        # 預測下一個價格
        prediction = model.predict(current_input.reshape(1, TIME_STEP, features_count), verbose=0)
        
        # 反轉標準化 (只針對 'Close' 價格，索引 0)
        prediction_scaled = np.zeros((1, features_count)) 
        prediction_scaled[0, 0] = prediction[0, 0]
        real_prediction = scaler.inverse_transform(prediction_scaled)[0, 0]
        future_predictions.append(real_prediction)
        
        # 更新輸入數據：用預測值替換掉第一天的數據
        new_feature_values = current_input[-1].copy() 
        new_feature_values[0] = prediction[0, 0] # 更新 'Close' 特徵
        
        current_input = np.vstack([current_input[1:], new_feature_values])
    
    # --- 繪圖與結果展示 ---
    predict_dates = [data.index[-1] + timedelta(days=i) for i in range(1, predict_days + 1)]
    fig = go.Figure()
    
    # 歷史 K 線圖
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='歷史K線'
    ))

    # 預測線
    fig.add_trace(go.Scatter(
        x=data.index.tolist()[-TIME_STEP:] + predict_dates, 
        y=data['Close'].tolist()[-TIME_STEP:] + future_predictions,
        mode='lines+markers',
        name=f'預測股價 ({predict_days}天)',
        line=dict(color='orange', width=3)
    ))
    
    # 加入布林帶
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='布林帶上軌'))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name='布林帶下軌'))

    fig.update_layout(title=f'{stock_ticker} 歷史股價與未來 {predict_days} 天預測',
                      xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True) 
    
    # --- 買賣點建議邏輯 ---
    st.markdown("### 🎯 近期最佳買入點與賣出點建議 (結合 LSTM 預測趨勢)")
    
    latest_close = data['Close'].iloc[-1]
    avg_future_price = np.mean(future_predictions)
    
    # 判斷預測走向：預測期內的價格變動百分比
    prediction_change_percent = (avg_future_price - latest_close) / latest_close * 100
    
    buy_advice = []
    sell_advice = []
    latest = data.iloc[-1]
    
    # --- 買入訊號 ---
    # 1. 強烈預測上漲 + RSI 不在超買區
    if prediction_change_percent >= 1.0 and 'RSI' in latest and latest['RSI'] < 70:
        buy_advice.append(f"📈 **LSTM 強力看漲 (+{prediction_change_percent:.2f}%)**: 預測未來股價有顯著上漲空間。")

    # 2. MACD 金叉 (MACD_Signal > 0 且 MACD > MACD_Signal) + 預測走勢向上
    if 'MACD' in latest and 'MACD_Signal' in latest and latest['MACD_Signal'] > 0 and latest['MACD'] > latest['MACD_Signal'] and prediction_change_percent > 0:
        buy_advice.append("💰 **MACD 金叉訊號** (MACD 線上穿訊號線): 動能轉強，結合預測趨勢向上。")
    
    # 3. 價格觸及布林帶下軌 (BB_Ratio 接近 0) + 預測反彈
    if 'BB_Ratio' in latest and latest['BB_Ratio'] < 0.1 and prediction_change_percent > 0.1: # 需預測至少微幅反彈
        buy_advice.append("📉 **布林帶下軌支撐**: 價格進入布林帶超賣區，預測有反彈機會。")

    # --- 賣出訊號 ---
    # 1. 強烈預測下跌 或 RSI 在極度超買區
    if prediction_change_percent <= -1.0 or ('RSI' in latest and latest['RSI'] > 75):
        sell_advice.append(f"📉 **LSTM 強力看跌 ({prediction_change_percent:.2f}%) / RSI 極度超買**: 預測下跌或 RSI 處於極度超買區。")

    # 2. MACD 死叉 + 預測走勢向下
    if 'MACD' in latest and 'MACD_Signal' in latest and latest['MACD'] < latest['MACD_Signal'] and prediction_change_percent < 0:
        sell_advice.append("🛑 **MACD 死叉訊號**: 短期動能向下突破訊號線，結合預測趨勢向下。")

    # 3. 價格觸及布林帶上軌 (BB_Ratio 接近 1)
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
st.title("股票數據預測與買賣點建議系統 (部署版) 🚀")
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
