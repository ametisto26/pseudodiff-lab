import numpy as np
import matplotlib.pyplot as plt

# 区間と格子
N = 1024
L = 20

x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

# 元の関数（高周波を少し混ぜる）
u = (
    np.exp(-x**2)
    + 0.3 * np.sin(20 * x)
)

# Fourier変換
u_hat = np.fft.fft(u)

# 周波数
xi = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# 擬微分作用素のシンボル
s = 2.0
symbol = (1 + xi**2)**(-s / 2)

# Fourier側で掛ける
Pu_hat = symbol * u_hat

# 逆変換
Pu = np.fft.ifft(Pu_hat).real

# 描画
plt.figure(figsize=(10, 5))
plt.plot(x, u, label="original")
plt.plot(x, Pu, label="after pseudo-differential operator")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Smoothing by Fourier Multiplier")
plt.grid()

plt.show()
