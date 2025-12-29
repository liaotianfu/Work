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
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CONFIG = {
    'window_size': 32,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'train_split': 0.8
}

# ---------------------------------------------------------
# 2. 數據準備
# ---------------------------------------------------------
print("Fetching Data...")
try:
    ticker = yf.Ticker("^IXIC")
    df = ticker.history(period="5y")
    if df.empty: raise ValueError("Empty Data")
except Exception as e:
    print(e); sys.exit()

close = df["Close"].values.astype(float)
dates = df.index.to_pydatetime()

# 切分 Train / Test
split_idx = int(len(close) * CONFIG['train_split'])

train_raw = close[:split_idx]
dates_train = dates[:split_idx]  # [新增] 為了畫圖

test_raw = close[split_idx:]
dates_test = dates[split_idx:]

print(f"Train Length: {len(train_raw)} | Test Length: {len(test_raw)}")

# ---------------------------------------------------------
# 3. 獨立 EMD 分解
# ---------------------------------------------------------
emd = EMD()

print("\nPerforming EMD on TRAINING data...")
imfs_train = emd.emd(train_raw)
num_imfs_train = imfs_train.shape[0]

print("Performing EMD on TESTING data...")
imfs_test = emd.emd(test_raw)
num_imfs_test = imfs_test.shape[0]

# 對齊數量
num_imfs = min(num_imfs_train, num_imfs_test)
print(f"Using first {num_imfs} IMFs for training.")

# ---------------------------------------------------------
# [新增] 視覺化 EMD 分解結果 (Debug 用)
# ---------------------------------------------------------
def plot_decomposition(raw_data, imfs, dates, title_prefix):
    """
    畫出原始數據 + 所有分解出來的 IMF
    """
    num_components = imfs.shape[0]
    total_plots = num_components + 1
    
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 2 * total_plots), sharex=True)
    if total_plots == 1: axes = [axes]
    
    # 畫原始數據
    axes[0].plot(dates, raw_data, color='black')
    axes[0].set_title(f"{title_prefix}: Original Data")
    axes[0].grid(True)
    
    # 畫每個 IMF
    for i in range(num_components):
        axes[i+1].plot(dates, imfs[i], color='tab:blue')
        axes[i+1].set_title(f"{title_prefix}: IMF {i+1}")
        axes[i+1].grid(True, alpha=0.5)
        
    plt.tight_layout()
    plt.show()

print("\nShowing EMD Decomposition Plots (Close window to continue)...")
# 1. 畫 Training Set 的分解
plot_decomposition(train_raw, imfs_train, dates_train, "TRAIN Set")

# 2. 畫 Testing Set 的分解
plot_decomposition(test_raw, imfs_test, dates_test, "TEST Set")


# ---------------------------------------------------------
# 4. 模型與 Dataset 定義
# ---------------------------------------------------------
def create_xy(input_array, tw):
    x_list, y_list = [], []
    L = len(input_array)
    if L <= tw: return np.empty((0, tw, 1)), np.empty((0, 1))
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------------------------------------
# 5. 訓練循環
# ---------------------------------------------------------
pred_len = len(test_raw) - CONFIG['window_size']
final_preds = np.zeros(pred_len)
debug_imf_data = [] # 儲存預測結果供後續畫圖

print("\n" + "="*30)
print("Start EMD-LSTM Training Loop")
print("="*30)

for i in range(num_imfs):
    print(f"Processing IMF {i+1}/{num_imfs} ...", end=" ")
    
    train_signal = imfs_train[i].reshape(-1, 1)
    test_signal = imfs_test[i].reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_signal)
    train_norm = scaler.transform(train_signal)
    test_norm = scaler.transform(test_signal)
    
    X_train, y_train = create_xy(train_norm, CONFIG['window_size'])
    X_test, y_test = create_xy(test_norm, CONFIG['window_size'])
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Skipped (Not enough data)")
        continue

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = LSTMModel(hidden_size=CONFIG['hidden_size'], num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(CONFIG['epochs']):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            
    model.eval()
    preds_list, actuals_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds_list.append(model(xb).cpu().numpy())
            actuals_list.append(yb.numpy())
    
    if len(preds_list) > 0:
        imf_pred = scaler.inverse_transform(np.vstack(preds_list)).flatten()
        imf_actual = scaler.inverse_transform(np.vstack(actuals_list)).flatten()
        
        current_len = min(len(final_preds), len(imf_pred))
        final_preds[:current_len] += imf_pred[:current_len]
        
        rmse_imf = np.sqrt(mean_squared_error(imf_actual[:current_len], imf_pred[:current_len]))
        debug_imf_data.append({
            "idx": i + 1,
            "actual": imf_actual[:current_len],
            "pred": imf_pred[:current_len],
            "rmse": rmse_imf
        })
        print(f"Done. RMSE: {rmse_imf:.4f}")

# ---------------------------------------------------------
# 6. 總體評估
# ---------------------------------------------------------
ground_truth = test_raw[CONFIG['window_size']:]
valid_len = min(len(final_preds), len(ground_truth))
y_pred = final_preds[:valid_len]
y_true = ground_truth[:valid_len]

mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f"\nFinal Combined MAPE: {mape:.2f}%")

# ---------------------------------------------------------
# 7. 視覺化預測結果 (Prediction vs Actual per IMF)
# ---------------------------------------------------------
plot_dates = dates_test[CONFIG['window_size'] : CONFIG['window_size'] + valid_len]
total_plots = len(debug_imf_data) + 1 
fig, axes = plt.subplots(total_plots, 1, figsize=(14, 4 * total_plots), sharex=True)
if total_plots == 1: axes = [axes]

axes[0].plot(plot_dates, y_true, label='Actual Price', color='black', linewidth=2)
axes[0].plot(plot_dates, y_pred, label='EMD-LSTM Prediction', color='red', linestyle='--')
axes[0].set_title(f"Total Prediction Results (MAPE: {mape:.2f}%)", fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True)

for i, data in enumerate(debug_imf_data):
    ax = axes[i + 1]
    p_len = len(plot_dates)
    ax.plot(plot_dates, data['actual'][:p_len], label=f'IMF {data["idx"]} Actual', color='tab:blue', alpha=0.7)
    ax.plot(plot_dates, data['pred'][:p_len], label=f'IMF {data["idx"]} Pred', color='tab:orange', linestyle='--', alpha=0.9)
    ax.set_title(f"IMF {data['idx']} Prediction Detail (RMSE: {data['rmse']:.4f})")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
