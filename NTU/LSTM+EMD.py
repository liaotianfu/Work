import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PyEMD import EMD   # 新增：EMD 分解

# 1. 設置裝置
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 數據準備與 EMD 分解
# -----------------------------------------------------------------
# 使用 yfinance 下載 NASDAQ 5 年 Close 歷史數據
ticker = yf.Ticker("^IXIC")
df = ticker.history(period="5y")  # 預設每日資料
close = df["Close"].values.astype(float)   # shape: (N,)

print("原始資料長度:", len(close))

# 使用 EMD 對收盤價做分解
emd = EMD()
imfs = emd.emd(close)  # imfs.shape -> (num_imf, N)
print("IMF 數量:", imfs.shape[0])

# 取最後一個 IMF 作為 LSTM 的訓練目標（通常是最低頻 / 趨勢分量）
last_imf = imfs[-1]    # shape: (N,)

# 轉成 (N, 1) 以符合後續 MinMaxScaler / LSTM 的格式
data = last_imf.reshape(-1, 1)

# 設定訓練窗口
train_window = 16

# 先切分數據，再進行標準化（只在訓練集上 fit）
train_size_idx = int(len(data) * 0.8)
train_data_raw = data[:train_size_idx]
test_data_raw = data[train_size_idx:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data_raw)

train_data_normalized = scaler.transform(train_data_raw)
test_data_normalized = scaler.transform(test_data_raw)

# 轉為 Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized)
test_data_normalized = torch.FloatTensor(test_data_normalized)

# 定義序列生成函數
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    if L <= tw:
        return []
    for i in range(L - tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 產生訓練與測試序列（都是基於最後 IMF）
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
test_inout_seq = create_inout_sequences(test_data_normalized, train_window)

print("Train sequences:", len(train_inout_seq))
print("Test sequences :", len(test_inout_seq))

# 3. 模型定義 (LSTM)
# -----------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 層：batch_first=True -> (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全連接層 (輸出層)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化 hidden state 和 cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)

        # 只取最後一個時間點的輸出
        out = out[:, -1, :]  # (batch, hidden_size)

        # 經過全連接層得到最終預測
        out = self.fc(out)   # (batch, 1)
        return out

# 4. 訓練模型
# -----------------------------------------------------------------
model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

print("Start Training...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0

    for seq, labels in train_inout_seq:
        seq = seq.to(device)
        labels = labels.to(device)

        # 調整維度為 (batch_size=1, seq_len=train_window, input_size=1)
        seq = seq.view(1, train_window, -1)

        optimizer.zero_grad()
        y_pred = model(seq)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_inout_seq)
    print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.8f}")

# 5. 模型評估與預測（針對最後 IMF）
# -----------------------------------------------------------------
model.eval()
test_predictions = []
test_actuals = []

print("Start Testing...")
with torch.no_grad():
    for seq, labels in test_inout_seq:
        seq = seq.to(device)
        seq = seq.view(1, train_window, -1)

        prediction = model(seq).item()
        test_predictions.append(prediction)
        test_actuals.append(labels.item())

# 反標準化，還原為 IMF 原始幅度
actual_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
actual_values = scaler.inverse_transform(np.array(test_actuals).reshape(-1, 1))

# 6. 結果可視化（只看最後 IMF 的預測）
# -----------------------------------------------------------------
plt.figure(figsize=(15, 6))
plt.plot(actual_values, label='Last IMF Actual', color='blue')
plt.plot(actual_predictions, label='Last IMF Prediction', color='red', linestyle='--')
plt.title(f'NASDAQ Last IMF Prediction using LSTM (Window={train_window})')
plt.xlabel('Time (Days in Test Set)')
plt.ylabel('IMF Amplitude')
plt.legend()
plt.grid(True)
plt.show()
