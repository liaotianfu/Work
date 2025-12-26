import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from PyEMD import EMD
from statsmodels.tsa.arima.model import ARIMA

# ----------------------------
# 1. 下載 NASDAQ 資料並做 EMD
# ----------------------------
ticker = yf.Ticker("^IXIC")
df = ticker.history(period="5y")          # 每日收盤
close = df["Close"].values.astype(float)  # shape: (N,)
dates = df.index                          # DatetimeIndex

print("原始資料長度:", len(close))

# EMD 分解
emd = EMD()
imfs = emd.emd(close)                     # imfs.shape -> (num_imf, N)
print("IMF 數量:", imfs.shape[0])

# 取最後一個 IMF（最低頻 / 趨勢）
last_imf = imfs[-1]                       # shape: (N,)

# ----------------------------
# 2. 切 train / test
# ----------------------------
train_ratio = 0.8
train_size = int(len(last_imf) * train_ratio)

train_imf = last_imf[:train_size]
test_imf = last_imf[train_size:]
train_dates = dates[:train_size]
test_dates = dates[train_size:]

print("Train len:", len(train_imf))
print("Test  len:", len(test_imf))

# ----------------------------
# 3. 配 ARIMA 模型（針對最後 IMF）
# ----------------------------
# ARIMA(p,d,q) 先給一組簡單的參數，例如 (2,1,2)
# 實際可用 AIC/BIC 或 grid search 做調優
order = (2, 1, 2)

model = ARIMA(train_imf, order=order)
model_fit = model.fit()
print(model_fit.summary())

# ----------------------------
# 4. 對測試區間做滾動預測
# ----------------------------
history = list(train_imf)   # 一開始的歷史資料
predictions = []

for t in range(len(test_imf)):
    # 以目前的 history 重建並重訓 ARIMA（rolling forecast）
    # 若想加速，可改用動態 forecast，而不是每步重訓
    model = ARIMA(history, order=order)
    model_fit = model.fit()
    
    # 預測下一步 (step=1)
    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)
    
    # 把真正的觀測值加入 history
    history.append(test_imf[t])

predictions = np.array(predictions)

# ----------------------------
# 5. 可視化結果（最後 IMF）
# ----------------------------
plt.figure(figsize=(14, 6))

# 畫出整段最後 IMF
plt.plot(dates, last_imf, label="Last IMF (All)", color="lightgray")

# 標示訓練 / 測試區段
plt.plot(train_dates, train_imf, label="Last IMF - Train", color="blue")
plt.plot(test_dates, test_imf, label="Last IMF - Test (Actual)", color="green")

# ARIMA 在測試區間的預測
plt.plot(test_dates, predictions, label=f"ARIMA{order} Forecast on Last IMF", 
         color="red", linestyle="--")

plt.title("NASDAQ ^IXIC - Last IMF with ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("IMF Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
