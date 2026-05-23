import numpy as np
import matplotlib.pyplot as plt

# 区間と格子
N = 1024
L = 20

x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

# 元の関数
u = (
    np.exp(-x**2)
    + 0.3 * np.sin(20 * x) # 高周波の関数を少し混ぜる
)

# Fourier変換
u_hat = np.fft.fft(u)

# 周波数
xi = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# 擬微分作用素のシンボル
s = 2.0
symbol = (1 + xi**2)**(-s / 2)

# シンボルをかける
Pu_hat = symbol * u_hat

# 逆変換
Pu = np.fft.ifft(Pu_hat).real

# 描画
# =========================
# 変数 x に対する関数の値の比較
# =========================
# u(x) と Pu(x) を同じ x に対して並べて描いている
# 「関数の形がどう変わったか」を見る

plt.figure(figsize=(10, 5))
plt.plot(x, u, label="original u(x)")   # 元の関数
plt.plot(x, Pu, label="transformed Pu(x)")  # 作用後の関数

plt.legend()
plt.xlabel("x")         # 入力変数
plt.ylabel("value")     # 関数値
plt.title("Comparison of u(x) and transformed function")
plt.grid()

# =========================
# 変数 ξ に対する変換後の値の比較
# =========================
# u(x) を変換して得られる関数 û(ξ) を扱っている
# 各 ξ ごとの|û(ξ)| を見ている
# 同様に Pu(x) を変換したものも比較している

plt.figure(figsize=(10, 5))

plt.plot(xi, np.abs(u_hat), label="before transformation |û(ξ)|")
plt.plot(xi, np.abs(Pu_hat), label="after transformation |P̂u(ξ)|")

plt.xlim(-50, 50)  # 見やすい範囲に制限（大きい |ξ| の挙動を確認するため）

plt.legend()
plt.xlabel("ξ (index for oscillation rate)")   # 振動の速さを表す変数
plt.ylabel("magnitude")                        # 各成分の強さ
plt.title("Comparison after change of representation")
plt.grid()

plt.show()
