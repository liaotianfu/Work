import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from vmdpy import VMD  # [必須安裝] pip install vmdpy
import sys

# ---------------------------------------------------------
# 1. 設置與參數
# ---------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'window_size': 32,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'train_split': 0.8,
    'K': 3,             # [VMD 參數] 強制分解成 5 層
    'alpha': 2000,      # [VMD 參數] 頻寬限制 (越大頻寬越窄)
    'tau': 0,           # [VMD 參數] 噪聲容忍度
}

# ---------------------------------------------------------
# 2. 數據準備
# ---------------------------------------------------------
print("Fetching Data...")
try:
    ticker = yf.Ticker("^IXIC")
    df = ticker.history(period="5y")
    if df.empty: raise ValueError("Empty Data")
except: sys.exit()

close = df["Close"].values.astype(float)
dates = df.index.to_pydatetime()

# 切分 Train / Test
split_idx = int(len(close) * CONFIG['train_split'])
train_raw = close[:split_idx]
test_raw = close[split_idx:]
dates_test = dates[split_idx:]

print(f"Train: {len(train_raw)} | Test: {len(test_raw)}")

# ---------------------------------------------------------
# 3. VMD 分解 (固定 K 層)
# ---------------------------------------------------------
def run_vmd(signal, K, alpha, tau):
    """
    VMD 分解函數
    return: u (K, N)  -- 所有的 Modes
    """
    # VMD 參數設定
    DC = 0             # 無直流分量
    init = 1           # 均勻初始化
    tol = 1e-7         # 收斂容忍度
    
    # 執行 VMD
    # u: modes, u_hat: 頻譜, omega: 中心頻率
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    return u

print(f"\nPerforming VMD (K={CONFIG['K']})...")

# Train VMD
imfs_train = run_vmd(train_raw, CONFIG['K'], CONFIG['alpha'], CONFIG['tau'])
print(f" -> Train decomposed shape: {imfs_train.shape}") # (5, Train_Len)

# Test VMD (獨立做，不洩漏)
imfs_test = run_vmd(test_raw, CONFIG['K'], CONFIG['alpha'], CONFIG['tau'])
print(f" -> Test decomposed shape: {imfs_test.shape}")   # (5, Test_Len)

# [重點] 這裡完全不需要做 min() 或 merge，因為 K 是固定的
num_imfs = CONFIG['K']

# ---------------------------------------------------------
# 4. 畫圖檢查 VMD 結果 (確保分解合理)
# ---------------------------------------------------------
def plot_vmd(signal, imfs, title):
    plt.figure(figsize=(10, 8))
    plt.subplot(len(imfs)+1, 1, 1)
    plt.plot(signal, 'k')
    plt.title(f"{title} - Original")
    
    for i in range(len(imfs)):
        plt.subplot(len(imfs)+1, 1, i+2)
        plt.plot(imfs[i], 'b')
        plt.title(f"Mode {i+1}")
    plt.tight_layout()
    plt.show()

# 如果想看圖可以把下面這行打開
plot_vmd(train_raw, imfs_train, "Train VMD")

# ---------------------------------------------------------
# 5. 模型與訓練 (LSTM) - 完全沿用之前的邏輯
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

# 準備容器
pred_len = len(test_raw) - CONFIG['window_size']
final_preds = np.zeros(pred_len)
debug_imf_data = []

print("\n" + "="*40)
print(f"Start VMD-LSTM Training (K={num_imfs})")
print("="*40)

for i in range(num_imfs):
    print(f"Processing Mode {i+1}/{num_imfs} ...", end=" ")
    
    # 1. 取資料
    train_signal = imfs_train[i].reshape(-1, 1)
    test_signal = imfs_test[i].reshape(-1, 1)
    
    # 2. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_signal)
    train_norm = scaler.transform(train_signal)
    test_norm = scaler.transform(test_signal)
    
    # 3. Dataset
    X_train, y_train = create_xy(train_norm, CONFIG['window_size'])
    X_test, y_test = create_xy(test_norm, CONFIG['window_size'])
    
    if len(X_train) == 0: continue

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 4. Train
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
            
    # 5. Predict
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
# 6. 結果與視覺化
# ---------------------------------------------------------
ground_truth = test_raw[CONFIG['window_size']:]
valid_len = min(len(final_preds), len(ground_truth))
y_pred = final_preds[:valid_len]
y_true = ground_truth[:valid_len]

mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f"\nFinal VMD-LSTM MAPE: {mape:.2f}%")

# 畫圖
plot_dates = dates_test[CONFIG['window_size'] : CONFIG['window_size'] + valid_len]
total_plots = len(debug_imf_data) + 1 
fig, axes = plt.subplots(total_plots, 1, figsize=(14, 3 * total_plots), sharex=True)
if total_plots == 1: axes = [axes]

axes[0].plot(plot_dates, y_true, label='Actual Price', color='black', linewidth=1.5)
axes[0].plot(plot_dates, y_pred, label='VMD-LSTM Prediction', color='red', linestyle='--')
axes[0].set_title(f"VMD Total Reconstruction (K={CONFIG['K']}, MAPE: {mape:.2f}%)")
axes[0].legend()
axes[0].grid(True)

for i, data in enumerate(debug_imf_data):
    ax = axes[i + 1]
    p_len = len(plot_dates)
    ax.plot(plot_dates, data['actual'][:p_len], label=f'Mode {data["idx"]} Actual', color='tab:blue', alpha=0.6)
    ax.plot(plot_dates, data['pred'][:p_len], label=f'Mode {data["idx"]} Pred', color='tab:orange', linestyle='--')
    ax.set_title(f"Mode {data['idx']} (RMSE: {data['rmse']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
