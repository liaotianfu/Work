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
from statsmodels.tsa.arima.model import ARIMA
import warnings

# 忽略 ARIMA 可能產生的 ConvergenceWarning (加速顯示)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. 設置與參數 (嚴格保持一致)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CONFIG = {
    'window_size': 32,      # 保持一致
    'hidden_size': 64,      # 保持一致
    'num_layers': 2,        # 保持一致
    'dropout': 0.2,         # 保持一致
    'batch_size': 32,       # 保持一致
    'epochs': 60,           # 保持一致
    'lr': 0.001,            # 保持一致
    'train_split': 0.8,     # 保持一致
    'arima_order': (5, 2, 2) # ARIMA 參數 (針對 Trend)
}

# ---------------------------------------------------------
# 2. 數據準備
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

# ---------------------------------------------------------
# 3. 定義 Dataset 與 LSTM Model (完全沿用)
# ---------------------------------------------------------
def create_xy(input_array, tw):
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
# 4. 混合訓練迴圈
# ---------------------------------------------------------
train_size_idx = int(len(close) * CONFIG['train_split'])

# 計算最終對齊的長度 (LSTM 需要 window_size 作為前置)
test_len_aligned = len(close) - train_size_idx - CONFIG['window_size']

final_preds = np.zeros(test_len_aligned)
final_actuals = np.zeros(test_len_aligned)

print("\n" + "="*60)
print(f"Start Hybrid Training: IMFs 0~{num_imfs-2} (LSTM) + IMF {num_imfs-1} (ARIMA)")
print("="*60)

for i in range(num_imfs):
    imf_data = imfs[i]
    
    # 切分 Train / Test (所有模型統一使用此切分點)
    train_raw = imf_data[:train_size_idx]
    test_raw = imf_data[train_size_idx:]
    
    # -----------------------------------------------------
    # 分支 A: 最後一個 IMF (Trend) -> 用 ARIMA
    # -----------------------------------------------------
    if i == num_imfs - 1:
        print(f"\nProcessing IMF {i+1} (Trend) -> [ARIMA]")
        
        # ARIMA 不需要 Scaler，直接用原始數值
        # 這裡做一次性 Forecast (預測未來 steps=len(test_raw))
        # 雖然沒有 LSTM 的 window 限制，但為了後面加總方便，我們稍後會裁切
        
        model_arima = ARIMA(train_raw, order=CONFIG['arima_order'])
        model_fit = model_arima.fit()
        
        # 預測整段測試區間
        forecast = model_fit.forecast(steps=len(test_raw))
        
        imf_pred = forecast
        imf_actual = test_raw
        
        # 關鍵對齊：LSTM 因為要吃 window，所以它的第一筆預測其實是 test_raw[window_size]
        # 為了讓 ARIMA 跟 LSTM 的時間軸對上，我們也要把 ARIMA 的前 window_size 筆丟掉
        # 這樣 final_preds[0] 才會對應到同一天
        imf_pred = imf_pred[CONFIG['window_size']:]
        imf_actual = imf_actual[CONFIG['window_size']:]
        
        print(f"  -> ARIMA RMSE: {np.sqrt(mean_squared_error(imf_actual, imf_pred)):.4f}")

    # -----------------------------------------------------
    # 分支 B: 其他 IMF (Noise/Cycle) -> 用 LSTM
    # -----------------------------------------------------
    else:
        print(f"\nProcessing IMF {i+1} (High Freq) -> [LSTM]")
        
        # 1. Reshape & Scale
        train_r = train_raw.reshape(-1, 1)
        test_r = test_raw.reshape(-1, 1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_r) # 嚴格遵守：只 Fit Train
        
        train_norm = scaler.transform(train_r)
        test_norm = scaler.transform(test_r)
        
        # 2. Create XY
        X_train, y_train = create_xy(train_norm, CONFIG['window_size'])
        X_test, y_test = create_xy(test_norm, CONFIG['window_size'])
        
        # 3. DataLoader
        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
        test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)
        
        # 4. Model Init
        model = LSTMModel(
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.MSELoss()
        
        # 5. Train
        model.train()
        for epoch in range(CONFIG['epochs']):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        
        # 6. Predict
        model.eval()
        preds_list = []
        actuals_list = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds_list.append(model(xb).cpu().numpy())
                actuals_list.append(yb.numpy())
        
        if len(preds_list) > 0:
            imf_pred_norm = np.vstack(preds_list)
            imf_actual_norm = np.vstack(actuals_list)
            
            imf_pred = scaler.inverse_transform(imf_pred_norm).flatten()
            imf_actual = scaler.inverse_transform(imf_actual_norm).flatten()
            print(f"  -> LSTM RMSE: {np.sqrt(mean_squared_error(imf_actual, imf_pred)):.4f}")
        else:
            print("  -> Warning: No prediction generated.")
            imf_pred = np.zeros(test_len_aligned)
            imf_actual = np.zeros(test_len_aligned)

    # -----------------------------------------------------
    # 加總 (Reconstruction)
    # -----------------------------------------------------
    # 防呆：取最小長度
    L = min(len(final_preds), len(imf_pred))
    final_preds[:L] += imf_pred[:L]
    final_actuals[:L] += imf_actual[:L]

# ---------------------------------------------------------
# 5. 最終評估
# ---------------------------------------------------------
# 裁切有效長度
valid_len = min(len(final_preds), len(final_actuals))
y_pred_final = final_preds[:valid_len]
y_true_final = final_actuals[:valid_len]

# 計算指標
rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
mae = mean_absolute_error(y_true_final, y_pred_final)
mape = np.mean(np.abs((y_true_final - y_pred_final) / y_true_final)) * 100

print("\n" + "="*30)
print(f"Hybrid Model Results (Compare with Pure LSTM)")
print("="*30)
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f} %")
print("="*30)

# ---------------------------------------------------------
# 6. 畫圖比較
# ---------------------------------------------------------
test_dates = dates[train_size_idx + CONFIG['window_size']:]
plot_dates = test_dates[:valid_len]

plt.figure(figsize=(15, 8))

# 1. 股價走勢
plt.subplot(2, 1, 1)
plt.plot(plot_dates, y_true_final, label='Actual Close', color='black', alpha=0.8)
plt.plot(plot_dates, y_pred_final, label='Hybrid (EMD-LSTM-ARIMA)', color='purple', linestyle='--')
plt.title(f"Hybrid Model Prediction (MAPE: {mape:.2f}%)")
plt.legend()
plt.grid(True)

# 2. 殘差
plt.subplot(2, 1, 2)
residuals = y_true_final - y_pred_final
plt.plot(plot_dates, residuals, color='blue', alpha=0.6, label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title("Prediction Residuals")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
