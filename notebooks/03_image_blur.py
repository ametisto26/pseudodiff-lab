import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# =========================
# 画像読込
# =========================

img = Image.open("sample1.png").convert("L")
# img = Image.open("sample2.png").convert("L")


u = np.array(img, dtype=float)


# =========================
# 2次元 FFT
# =========================

u_hat = np.fft.fft2(u)


# =========================
# 周波数変数
# =========================

ny, nx = u.shape

xi_x = np.fft.fftfreq(nx)
xi_y = np.fft.fftfreq(ny)

XI_X, XI_Y = np.meshgrid(xi_x, xi_y)

r2 = XI_X**2 + XI_Y**2


# =========================
# Gaussian blur
# =========================

sigma = 100

symbol = np.exp(-sigma * r2)
# symbol = 1 - np.exp(-sigma * r2)

# =========================
# multiplier 適用
# =========================

Pu_hat = symbol * u_hat

Pu = np.fft.ifft2(Pu_hat).real


# =========================
# 描画
# =========================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(u, cmap="gray")
plt.title("original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(Pu, cmap="gray")
plt.title("blurred")
plt.axis("off")

plt.imshow(np.log1p(np.abs(np.fft.fftshift(u_hat))), cmap="gray")

plt.show()
