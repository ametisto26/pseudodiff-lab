import numpy as np
import matplotlib.pyplot as plt

n = 256

x = np.arange(n)
y = np.arange(n)

X, Y = np.meshgrid(x, y)

u = ((X // 8 + Y // 8) % 2).astype(float)

plt.imshow(u, cmap="gray")
plt.axis("off")

plt.show()
