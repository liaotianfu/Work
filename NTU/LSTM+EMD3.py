import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from PyEMD import EMD
from torch.utils.data import Dataset, DataLoader
import sys

# ---------------------------------------------------------
# 1. 設置與參數
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 參數設定 (與你之前的設定保持一致)
CONFIG = {
    'window_size': 32,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 32,
    'epochs': 60,       # 針對每個 IMF 訓練 60 輪
    'lr': 0.001,
    'train_split': 0.8
}

# ---------------------------------------------------------
# 2. 數據準備與 EMD 分解
# ---------------------------------------------------------
print("Fetching Data...")
ticker = yf.Ticker("^IXIC")
df = ticker.history(period="5y")
close = df["Close"].values.astype(float)
dates = df.index.to_pydatetime()

print(f"Original Data Length: {len(close)}")

# EMD 分解
print("Performing EMD Decomposition...")
emd = EMD()
imfs = emd.emd(close)  # shape: (num_imfs, N)
num_imfs = imfs.shape[0]

print(f"Decomposed into {num_imfs} IMFs.")

# 檢查重建誤差 (確認 sum(imfs) 是否等於 close)
reconstructed_check = np.sum(imfs, axis=0)
err = np.sum(np.abs(close - reconstructed_check))
print(f"EMD Reconstruction Error (Check): {err:.4f}")

# ---------------------------------------------------------
# 3. 定義 Dataset 與 Model (共用)
# ---------------------------------------------------------
def create_xy(input_array, tw):
    """
    input_array: (N, 1)
    return: X(samples, tw, 1), y(samples, 1)
    """
    x_list, y_list = [], []
    L = len(input_array)
    if L <= tw:
        return np.empty((0, tw, 1)), np.empty((0, 1))
    for i in range(L - tw):
        x_list.append(input_array[i:i+tw])
        y_list.append(input_array[i+tw])
    return np.array(x_list), np.array(y_list)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

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
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ---------------------------------------------------------
# 4. 針對每個 IMF 進行訓練與預測 (Loop)
# ---------------------------------------------------------
# 用來儲存所有 IMF 的預測結果，最後要加總
total_prediction_sum = None 
total_actual_sum = None

# 我們需要對齊時間軸，因為 create_xy 會吃掉前面 window_size 的資料
train_size_idx = int(len(close) * CONFIG['train_split'])
test_len_after_window = len(close) - train_size_idx - CONFIG['window_size']

# 初始化加總陣列 (大小為測試集長度)
final_preds = np.zeros(test_len_after_window)
final_actuals = np.zeros(test_len_after_window)

print("\n" + "="*50)
print(f"Start EMD-LSTM Ensemble Training ({num_imfs} Models)")
print("="*50)

for i in range(num_imfs):
    print(f"\nProcessing IMF {i+1}/{num_imfs} ...")
    
    # 1. 準備當前 IMF 的資料
    imf_data = imfs[i].reshape(-1, 1)
    
    # 2. 切分 Train / Test
    # 嚴格遵守：先切分，再 Scaling
    train_raw = imf_data[:train_size_idx]
    test_raw = imf_data[train_size_idx:]
    
    # 3. Scaling (每個 IMF 的震幅不同，必須獨立 Scale)
    # IMF 通常在 0 上下震盪，使用 MinMaxScaler(0, 1) 或 StandardScaler 均可
    # 這裡沿用你的 (0, 1) 設定
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)
    
    train_norm = scaler.transform(train_raw)
    test_norm = scaler.transform(test_raw)
    
    # 4. 製作序列 (X, y)
    X_train, y_train = create_xy(train_norm, CONFIG['window_size'])
    X_test, y_test = create_xy(test_norm, CONFIG['window_size'])
    
    # 建立 DataLoader
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 5. 初始化模型 (每個 IMF 用一個新模型)
    model = LSTMModel(
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 6. 訓練
    model.train()
    for epoch in range(CONFIG['epochs']):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    
    # 7. 預測
    model.eval()
    preds_list = []
    actuals_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p = model(xb)
            preds_list.append(p.cpu().numpy())
            actuals_list.append(yb.numpy())
            
    # 串接
    if len(preds_list) > 0:
        imf_pred_norm = np.vstack(preds_list)
        imf_actual_norm = np.vstack(actuals_list)
        
        # 反歸一化
        imf_pred = scaler.inverse_transform(imf_pred_norm).flatten()
        imf_actual = scaler.inverse_transform(imf_actual_norm).flatten()
        
        # 8. 加總到最終結果
        # 確保長度一致 (有些 batch 可能導致長度微調，取 min 長度)
        length = min(len(final_preds), len(imf_pred))
        final_preds[:length] += imf_pred[:length]
        final_actuals[:length] += imf_actual[:length]
        
        print(f"  -> IMF {i+1} Prediction Finished. (RMSE: {np.sqrt(mean_squared_error(imf_actual[:length], imf_pred[:length])):.4f})")
    else:
        print("  -> Warning: No test data generated (Window size too large relative to test set?)")

# ---------------------------------------------------------
# 5. 最終評估
# ---------------------------------------------------------
# 裁切掉多餘的長度 (如果有)
valid_len = min(len(final_preds), len(final_actuals))
y_pred_final = final_preds[:valid_len]
y_true_final = final_actuals[:valid_len]

# 評估指標
rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
mae = mean_absolute_error(y_true_final, y_pred_final)
mape = np.mean(np.abs((y_true_final - y_pred_final) / y_true_final)) * 100

print("\n" + "="*30)
print(f"EMD-LSTM Ensemble Results")
print("="*30)
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f} %")
print("="*30)

# ---------------------------------------------------------
# 6. 視覺化
# ---------------------------------------------------------
test_dates = dates[train_size_idx + CONFIG['window_size']:]
# 確保日期長度匹配
plot_dates = test_dates[:len(y_pred_final)]

plt.figure(figsize=(15, 8))

# 主圖
plt.subplot(2, 1, 1)
plt.plot(plot_dates, y_true_final, label='Actual Close', color='black', alpha=0.8)
plt.plot(plot_dates, y_pred_final, label='EMD-LSTM Prediction', color='red', linestyle='--')
plt.title(f"EMD-LSTM Reconstruction Prediction (MAPE: {mape:.2f}%)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# 殘差圖
plt.subplot(2, 1, 2)
residuals = y_true_final - y_pred_final
plt.plot(plot_dates, residuals, color='blue', alpha=0.6, label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title("Prediction Residuals")
plt.xlabel("Date")
plt.ylabel("Error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
