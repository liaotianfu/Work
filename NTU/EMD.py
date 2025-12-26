import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

def download_nasdaq_close(period="5y"):
    """
    使用 yfinance 下載 NASDAQ (^IXIC) 收盤價
    """
    ticker = yf.Ticker("^IXIC")
    df = ticker.history(period=period)
    close = df["Close"].values.astype(float)  # shape: (N,)
    dates = df.index.to_pydatetime()         # 用來畫圖的日期
    return dates, close

def emd_decompose(signal):
    """
    對一維時間序列做 EMD 分解
    回傳 imfs (num_imf, N) 和 residue (N,)
    """
    N = len(signal)
    t = np.arange(N)  # 這裡用等距索引當作時間軸

    emd = EMD()
    imfs = emd.emd(signal, t)  # imfs.shape -> (num_imf, N)

    # 殘差/趨勢：原訊號減去所有 IMF 相加
    residue = signal - imfs.sum(axis=0)
    return imfs, residue

def plot_emd_result(dates, signal, imfs, residue):
    """
    繪製原始 NASDAQ 收盤價、各 IMF 與趨勢殘差
    """
    num_imf = imfs.shape[0]
    total_plots = num_imf + 2  # 原始訊號 + 各 IMF + residue

    plt.figure(figsize=(12, 2 * total_plots))

    # 1) 原始訊號
    ax = plt.subplot(total_plots, 1, 1)
    ax.plot(dates, signal, label="Close")
    ax.set_title("NASDAQ ^IXIC Close (Original)")
    ax.grid(True)

    # 2) 各 IMF
    for i in range(num_imf):
        ax = plt.subplot(total_plots, 1, i + 2)
        ax.plot(dates, imfs[i], label=f"IMF {i+1}")
        ax.set_ylabel(f"IMF {i+1}")
        ax.grid(True)

    # 3) 殘差/趨勢
    ax = plt.subplot(total_plots, 1, total_plots)
    ax.plot(dates, residue, label="Residue / Trend", color="red")
    ax.set_ylabel("Residue")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. 下載 5 年 NASDAQ 收盤價
    dates, close = download_nasdaq_close(period="5y")

    # 2. 做 EMD 分解
    imfs, residue = emd_decompose(close)
    print("IMF 數量:", imfs.shape[0])

    # 3. 畫圖檢查結果
    plot_emd_result(dates, close, imfs, residue)
