import numpy as np
import matplotlib.pyplot as plt


# =========================
# 格子生成
# =========================

N = 2048
L = 100

x = np.linspace(-L/2, L/2, N, endpoint=False)

dx = x[1] - x[0]

xi = 2 * np.pi * np.fft.fftfreq(N, d=dx)


# =========================
# 初期値
# =========================
#
# Gaussian wave packet
#

xi0 = 8.0

u0 = np.exp(-x**2) #* np.exp(1j * xi0 * x)

u0_hat = np.fft.fft(u0)


# =========================
# アニメーション準備
# =========================

plt.ion()

fig, ax = plt.subplots(figsize=(10, 5))

line, = ax.plot(x, np.abs(u0))

ax.set_xlim(-20, 20)
ax.set_ylim(0, 1.2)

ax.set_xlabel("x")
ax.set_ylabel("|u(x)|")

ax.grid()


# =========================
# 時間発展
# =========================

times = np.linspace(0, 20.0, 2000)

for t in times:

    # Schrödinger propagator
    propagator = np.exp(-1j * t * xi**2)

    # Fourier変換後の表示で時間発展
    u_hat = propagator * u0_hat

    # 元の表示に戻す
    u = np.fft.ifft(u_hat)

    # グラフ更新
    line.set_ydata(np.abs(u))

    ax.set_title(f"Schrödinger evolution  t = {t:.2f}")

    plt.pause(0.03)


plt.ioff()
plt.show()
