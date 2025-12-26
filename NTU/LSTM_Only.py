import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# 1. 設置裝置
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. 數據準備 (無 EMD，直接使用 Close)
# ---------------------------------------------------------
print("Fetching Data...")
ticker = yf.Ticker("^IXIC")
df = ticker.history(period="5y")  # 每日資料
close = df["Close"].values.astype(float)   # shape: (N,)
dates = df.index.to_pydatetime()

print("原始資料長度:", len(close))

# 為了配合你的原始架構，將數據 reshape
data = close.reshape(-1, 1)

train_window = 32  # 保持與你原本設定一致

# --- 關鍵修正：先切 train/test，再做 scaler fit ---
train_size_idx = int(len(data) * 0.8)

train_data_raw = data[:train_size_idx]
test_data_raw = data[train_size_idx:]

scaler = MinMaxScaler(feature_range=(0, 1))
# 只用 Training Data 來 Fit，防止未來的最高/最低價洩漏到訓練集
scaler.fit(train_data_raw)

train_data_normalized = scaler.transform(train_data_raw)
test_data_normalized = scaler.transform(test_data_raw)

# ---------------------------------------------------------
# 3. 建 Dataset / DataLoader (完全沿用你的函數)
# ---------------------------------------------------------
def create_xy(input_array, tw):
    """
    input_array: ndarray, shape (N, 1)
    return X: (num_samples, tw, 1), y: (num_samples, 1)
    """
    x_list, y_list = [], []
    L = len(input_array)
    if L <= tw:
        return np.empty((0, tw, 1)), np.empty((0, 1))
    for i in range(L - tw):
        x_list.append(input_array[i:i+tw])
        y_list.append(input_array[i+tw])
    return np.array(x_list), np.array(y_list)

X_train, y_train = create_xy(train_data_normalized, train_window)
X_test, y_test = create_xy(test_data_normalized, train_window)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Train samples:", len(train_dataset))
print("Test samples :", len(test_dataset))

# ---------------------------------------------------------
# 4. LSTM 模型 (完全沿用你的架構)
# ---------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))   # (batch, seq_len, hidden)
        out = out[:, -1, :]               # (batch, hidden)
        out = self.fc(out)                # (batch, 1)
        return out

# 保持一致的參數設定
model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 60  # 保持一致

# ---------------------------------------------------------
# 5. 訓練
# ---------------------------------------------------------
print("Start Training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for seq_batch, labels_batch in train_loader:
        seq_batch = seq_batch.to(device)      # (B, T, 1)
        labels_batch = labels_batch.to(device)  # (B, 1)

        optimizer.zero_grad()
        y_pred = model(seq_batch)
        loss = criterion(y_pred, labels_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * seq_batch.size(0)

    avg_loss = epoch_loss / len(train_dataset)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.8f}")

# ---------------------------------------------------------
# 6. 評估 / 預測
# ---------------------------------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for seq_batch, labels_batch in test_loader:
        seq_batch = seq_batch.to(device)
        preds = model(seq_batch)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels_batch.cpu().numpy())

test_predictions = np.vstack(all_preds)  # (N_test, 1)
test_actuals = np.vstack(all_labels)     # (N_test, 1)

# 反標準化
actual_predictions = scaler.inverse_transform(test_predictions)
actual_values = scaler.inverse_transform(test_actuals)

# ---------------------------------------------------------
# 7. 誤差計算 (Metrics)
# ---------------------------------------------------------
y_true = actual_values.flatten()
y_pred = actual_predictions.flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n" + "="*30)
print(f"Vanilla LSTM Performance (No EMD)")
print("="*30)
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f} %")
print("="*30)

# ---------------------------------------------------------
# 8. 畫圖 (Standard Price Prediction)
# ---------------------------------------------------------
# 對齊日期 (create_xy 會吃掉 window size)
test_dates = dates[train_size_idx + train_window:]
plot_len = min(len(test_dates), len(actual_values))

plt.figure(figsize=(15, 6))
plt.plot(test_dates[:plot_len], actual_values[:plot_len], label='Actual Close', color='black')
plt.plot(test_dates[:plot_len], actual_predictions[:plot_len], label='LSTM Prediction', color='red', linestyle='--')
plt.title(f'NASDAQ Prediction using Vanilla LSTM (Window={train_window})')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 畫殘差
plt.figure(figsize=(15, 4))
residuals = actual_values[:plot_len] - actual_predictions[:plot_len]
plt.plot(test_dates[:plot_len], residuals, label='Residuals', color='blue', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Prediction Residuals')
plt.grid(True)
plt.show()
